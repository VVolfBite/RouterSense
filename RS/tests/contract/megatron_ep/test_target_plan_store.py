from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.target_planning import TargetLayerPreparedJointPlan, TargetPlanKey, TargetPlanStore
from rs.scheduling.contracts import LogicalSchedulePlan
from rs.runtime.guards import RouterSenseInvariantError


def _plan() -> TargetLayerPreparedJointPlan:
    return TargetLayerPreparedJointPlan(
        source_layer_id="0",
        target_layer_id="1",
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        h1_prediction_digest="h1",
        h2_prediction_digest="h2",
        target_problem_digest="tp",
        logical_plan=LogicalSchedulePlan(policy_name="u", waves=(), diagnostics={}),
        logical_plan_digest="ld",
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
    assert claimed.logical_plan_digest == "ld"
    assert store.peek(key) is None
    consumed = store.consume_once(key)
    assert consumed.logical_plan_digest == "ld"
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
    assert claimed.logical_plan_digest == "ld"
    assert store.get_state_record(key).state == "CLAIMED"
    store.bind(key, bound_owner="executor")
    assert store.get_state_record(key).state == "BOUND"
    store.start_execution(key, execution_origin="executor_start", claim_owner="runtime")
    state = store.get_state_record(key)
    assert state.state == "EXECUTING"
    assert state.execution_origin == "executor_start"
    completed = store.complete(key, execution_origin="executor_complete")
    assert completed.logical_plan_digest == "ld"
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
