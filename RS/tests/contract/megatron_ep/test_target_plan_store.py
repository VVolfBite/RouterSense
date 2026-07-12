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
    consumed = store.consume_once(key)
    assert consumed.logical_plan_digest == "ld"
    with pytest.raises(RouterSenseInvariantError):
        store.consume_once(key)


def test_target_plan_store_cancel_and_invalidate() -> None:
    store = TargetPlanStore()
    key = TargetPlanKey("run", 1, "mb", "1")
    store.put(key, _plan())
    store.cancel(key)
    assert store.peek(key) is None
    store.put(key, _plan())
    store.invalidate(key)
    assert store.peek(key) is None

