from __future__ import annotations

from rs.core.contracts import WindowPlan
from rs.planning.api import to_logical_plan
from rs.runtime.online.megatron_ep.target_planning import TargetLayerPreparedJointPlan, reconcile_target_plan
from rs.scheduling.validation import stable_hash


def _prepared(h1=((0, 2), (1, 0))) -> TargetLayerPreparedJointPlan:
    window_plan = WindowPlan(
        planner_id="u",
        planner_family="joint",
        request_digest="req",
        waves=(),
        metadata={"policy_name": "u"},
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
        logical_payload_digest=stable_hash(logical_plan.to_dict()),
        policy="u",
        weights={},
        bucket_contract_digest="bucket",
        topology_digest="topo",
        h1_rows=h1,
        derived_p1_rows=((0, 1), (2, 0)),
        h2_rows=((0, 1), (1, 0)),
        created_at_ns=1,
        ready_at_ns=2,
    )


def test_reconcile_exact_match() -> None:
    outcome = reconcile_target_plan(prepared_plan=_prepared(), actual_p0_rows=((0, 2), (1, 0)))
    assert outcome.status == "exact"


def test_reconcile_repairable() -> None:
    outcome = reconcile_target_plan(prepared_plan=_prepared(), actual_p0_rows=((0, 4), (1, 0)))
    assert outcome.status == "repaired"
    assert outcome.resized_edges == 1


def test_reconcile_reject() -> None:
    outcome = reconcile_target_plan(prepared_plan=_prepared(h1=((0, 2), (0, 0))), actual_p0_rows=((0, 0), (5, 0)))
    assert outcome.status == "rejected"
