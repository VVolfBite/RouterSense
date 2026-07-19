from __future__ import annotations

import pytest

from rs.core.contracts import WindowPlan
from rs.planning.api import to_logical_plan
from rs.runtime.online.megatron_ep.target_planning import TargetLayerPreparedJointPlan, TargetPlanKey, TargetPlanStore
from rs.runtime.online.megatron_ep.target_planning.contracts import PreparationToken
from rs.runtime.guards import RouterSenseInvariantError
from rs.scheduling.validation import stable_hash


def _plan() -> TargetLayerPreparedJointPlan:
    window_plan = WindowPlan(
        planner_id="u",
        planner_family="joint",
        request_digest="req",
        waves=(),
        metadata={"legacy_policy_name": "u"},
    )
    logical_plan = to_logical_plan(window_plan)
    return TargetLayerPreparedJointPlan(
        source_layer_id="0",
        target_layer_id="1",
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        h1_prediction_digest="h1",
        h2_prediction_digest="h2",
        target_problem_digest="tp",
        window_plan=window_plan,
        logical_plan=logical_plan,
        logical_plan_digest=window_plan.semantic_digest(),
        legacy_logical_plan_digest=stable_hash(logical_plan.to_dict()),
        policy="u",
        weights={},
        bucket_contract_digest="bucket",
        topology_digest="topo",
        h1_rows=((0, 2), (1, 0)),
        derived_p1_rows=((0, 1), (2, 0)),
        h2_rows=((0, 3), (4, 0)),
        created_at_ns=1,
        ready_at_ns=2,
    )


def test_target_plan_store_put_peek_consume_once() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    plan = _plan()
    store.put(key, plan)
    assert store.peek(key) is plan
    claimed = store.claim_for_reconciliation(key)
    assert claimed.logical_plan_digest == plan.logical_plan_digest
    assert store.peek(key) is None
    consumed = store.consume_once(key)
    assert consumed.logical_plan_digest == plan.logical_plan_digest
    with pytest.raises(RouterSenseInvariantError):
        store.consume_once(key)


def test_target_plan_store_close_key_if_unclaimed_creates_terminal_tombstone() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    store.close_key_if_unclaimed(key, final_status="EXPIRED", execution_origin="too_late_no_effect")
    terminal = store.get_terminal_record(key)
    assert terminal is not None
    assert terminal.final_status == "EXPIRED"
    with pytest.raises(RouterSenseInvariantError):
        store.put(key, _plan())


def test_target_plan_store_cancel_and_invalidate() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    store.put(key, _plan())
    store.cancel(key)
    assert store.peek(key) is None
    assert store.get_terminal_record(key) is not None
    with pytest.raises(RouterSenseInvariantError):
        store.put(key, _plan())


def test_target_plan_store_formal_state_transitions() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    plan = _plan()
    store.publish_logical(key, plan)
    assert store.get_state_record(key).state == "LOGICAL_READY"
    claimed = store.claim(key, claim_owner="runtime")
    assert claimed.logical_plan_digest == plan.logical_plan_digest
    assert store.get_state_record(key).state == "CLAIMED"
    store.bind(key, bound_owner="executor")
    assert store.get_state_record(key).state == "BOUND"
    store.start_execution(key, execution_origin="executor_start", claim_owner="runtime")
    state = store.get_state_record(key)
    assert state.state == "EXECUTING"
    assert state.execution_origin == "executor_start"
    completed = store.complete(key, execution_origin="executor_complete")
    assert completed.logical_plan_digest == plan.logical_plan_digest
    terminal = store.get_terminal_record(key)
    assert terminal is not None
    assert terminal.final_status == "COMPLETED"


def test_target_plan_store_fail_after_claim_records_failed_terminal() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    store.publish_logical(key, _plan())
    store.claim(key, claim_owner="runtime")
    store.fail(key, execution_origin="compile_failed")
    terminal = store.get_terminal_record(key)
    assert terminal is not None
    assert terminal.final_status == "FAILED"


def test_target_plan_store_rejects_illegal_transitions() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    store.publish_logical(key, _plan())
    with pytest.raises(RouterSenseInvariantError):
        store.start_execution(key, execution_origin="bad_direct_start")
    with pytest.raises(RouterSenseInvariantError):
        store.complete(key, execution_origin="bad_direct_complete")
    store.claim(key, claim_owner="runtime")
    with pytest.raises(RouterSenseInvariantError):
        store.complete(key, execution_origin="bad_claim_complete")


def test_target_plan_store_publish_if_current_is_atomic_and_structured() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    token = PreparationToken(
        service_session_id=1,
        forward_generation=1,
        target_key=key,
        task_version=2,
        publish_sequence=3,
    )
    assert store.register_expected_publication(token) is True
    published = store.publish_if_current(token=token, plan=_plan())
    assert published.status == "PUBLISHED"
    assert store.peek(key) is not None
    stale = store.publish_if_current(
        token=PreparationToken(
            service_session_id=1,
            forward_generation=1,
            target_key=key,
            task_version=1,
            publish_sequence=3,
        ),
        plan=_plan(),
    )
    assert stale.status in {"TERMINAL", "STALE_TOKEN", "CONFLICTING_PLAN", "ALREADY_PUBLISHED_SAME"}


def _plan_for_epoch(epoch: int) -> TargetLayerPreparedJointPlan:
    plan = _plan()
    payload = plan.to_dict()
    payload["forward_epoch"] = int(epoch)
    return TargetLayerPreparedJointPlan.from_dict(payload)


def test_target_plan_store_cleanup_before_generation_clears_all_older_tokens() -> None:
    store = TargetPlanStore()
    key1 = TargetPlanKey("run", 1, "mb", "1")
    key2 = TargetPlanKey("run", 2, "mb", "1")
    key3 = TargetPlanKey("run", 3, "mb", "1")
    assert store.register_expected_publication(
        PreparationToken(service_session_id=1, forward_generation=1, target_key=key1, task_version=1, publish_sequence=1)
    ) is True
    assert store.register_expected_publication(
        PreparationToken(service_session_id=1, forward_generation=2, target_key=key2, task_version=1, publish_sequence=1)
    ) is True
    assert store.register_expected_publication(
        PreparationToken(service_session_id=1, forward_generation=3, target_key=key3, task_version=1, publish_sequence=1)
    ) is True
    store.cleanup_before_generation(run_id="run", microbatch_id="mb", current_generation=3)
    assert store.publish_if_current(
        token=PreparationToken(service_session_id=1, forward_generation=1, target_key=key1, task_version=1, publish_sequence=1),
        plan=_plan_for_epoch(1),
    ).status == "EXPIRED_GENERATION"
    assert store.publish_if_current(
        token=PreparationToken(service_session_id=1, forward_generation=2, target_key=key2, task_version=1, publish_sequence=1),
        plan=_plan_for_epoch(2),
    ).status == "EXPIRED_GENERATION"
    result = store.publish_if_current(
        token=PreparationToken(service_session_id=1, forward_generation=3, target_key=key3, task_version=1, publish_sequence=1),
        plan=_plan_for_epoch(3),
    )
    assert result.status == "PUBLISHED"


def test_target_plan_store_generation_floor_rejects_future_register_of_old_generation() -> None:
    store = TargetPlanStore()
    key1 = TargetPlanKey("run", 1, "mb", "1")
    key3 = TargetPlanKey("run", 3, "mb", "1")
    store.cleanup_before_generation(run_id="run", microbatch_id="mb", current_generation=3)
    assert store.register_expected_publication(
        PreparationToken(service_session_id=1, forward_generation=1, target_key=key1, task_version=1, publish_sequence=1)
    ) is False
    assert store.publish_if_current(
        token=PreparationToken(service_session_id=1, forward_generation=1, target_key=key1, task_version=1, publish_sequence=1),
        plan=_plan_for_epoch(1),
    ).status == "EXPIRED_GENERATION"
    assert store.register_expected_publication(
        PreparationToken(service_session_id=1, forward_generation=3, target_key=key3, task_version=1, publish_sequence=1)
    ) is True


def test_target_plan_store_does_not_let_older_token_override_newer_token() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    newer = PreparationToken(service_session_id=2, forward_generation=1, target_key=key, task_version=10, publish_sequence=10)
    older = PreparationToken(service_session_id=1, forward_generation=1, target_key=key, task_version=1, publish_sequence=1)
    assert store.register_expected_publication(newer) is True
    assert store.register_expected_publication(older) is False
    result = store.publish_if_current(token=newer, plan=_plan())
    assert result.status == "PUBLISHED"


def test_target_plan_store_same_token_is_idempotent() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    token = PreparationToken(service_session_id=1, forward_generation=1, target_key=key, task_version=1, publish_sequence=1)
    assert store.register_expected_publication(token) is True
    assert store.register_expected_publication(token) is True
