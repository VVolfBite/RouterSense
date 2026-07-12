from __future__ import annotations

from rs.runtime.online.megatron_ep.target_planning import (
    CurrentWindowJointPlan,
    PreparedPriorityHint,
    ProvisionalExecutionPlan,
    TargetLayerPreparedJointPlan,
)
from rs.scheduling.contracts import LogicalSchedulePlan


def _plan(name: str = "policy") -> LogicalSchedulePlan:
    return LogicalSchedulePlan(policy_name=name, waves=(), diagnostics={})


def test_current_window_joint_plan_contract() -> None:
    plan = CurrentWindowJointPlan(
        source_layer_id="0",
        execution_layer_id="0",
        forecast_target_layer_id="1",
        logical_plan=_plan(),
        logical_plan_digest="abcd",
        actual_p0_rows=((0, 2), (1, 0)),
        inferred_p1_rows=((0, 1), (2, 0)),
        forecast_h1_rows=((0, 3), (4, 0)),
        created_at_ns=1,
    )
    payload = plan.to_dict()
    assert payload["execution_layer_id"] == "0"
    assert payload["forecast_target_layer_id"] == "1"


def test_target_prepared_plan_contract() -> None:
    plan = TargetLayerPreparedJointPlan(
        source_layer_id="0",
        target_layer_id="1",
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        h1_prediction_digest="h1",
        h2_prediction_digest="h2",
        target_problem_digest="tp",
        logical_plan=_plan("u"),
        logical_plan_digest="ld",
        policy="u_pred",
        weights={"prediction_weight": 0.3},
        bucket_contract_digest="bucket",
        topology_digest="topo",
        h1_rows=((0, 2), (1, 0)),
        derived_p1_rows=((0, 1), (2, 0)),
        h2_rows=((0, 1), (3, 0)),
        created_at_ns=1,
        ready_at_ns=2,
    )
    payload = plan.to_dict()
    assert payload["target_layer_id"] == "1"
    assert payload["plan_origin"] == "target_prepared"
    assert payload["selected_variant"] == "raw_u"
    assert payload["paired_b_logical_plan_digest"] == ""


def test_prepared_priority_hint_is_not_target_plan() -> None:
    hint = PreparedPriorityHint(
        source_layer_id="0",
        target_layer_id="1",
        priority_digest="digest",
        preferred_edges=(("P0", 0, 1),),
        created_at_ns=1,
    )
    provisional = ProvisionalExecutionPlan(
        target_layer_id="1",
        plan_origin="provisional",
        plan_version=0,
        parent_plan_version=0,
        logical_plan=_plan("prov"),
        logical_plan_digest="prov",
        created_at_ns=1,
        execution_started_at_ns=2,
    )
    assert hint.to_dict()["priority_digest"] == "digest"
    assert provisional.to_dict()["plan_origin"] == "provisional"
