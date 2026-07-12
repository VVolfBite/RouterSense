from __future__ import annotations

from rs.runtime.online.megatron_ep.target_planning.planner_service import (
    TargetLayerPlannerMetrics,
    TargetLayerPlannerService,
    TargetLayerPlanningRequest,
)
from rs.runtime.online.megatron_ep.target_planning.store import TargetPlanStore
from rs.scheduling.unified_interface import PolicyOptions


def _request(*, safe_projection_mode: str) -> TargetLayerPlanningRequest:
    return TargetLayerPlanningRequest(
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
        current_p0_rows=((0, 2, 5), (3, 0, 3), (1, 5, 0)),
        previous_p0_rows=((0, 1, 4), (2, 0, 2), (1, 4, 0)),
        predictor_name="copy_current_dispatch",
        policy_id="U_barrier_criticality_global_matching",
        raw_u_policy_id="U_barrier_criticality_global_matching",
        paired_b_policy_id="B_barrier_criticality_core_independent",
        safe_projection_mode=safe_projection_mode,
        group_size=3,
        bucket_rows=0,
        policy_options=PolicyOptions(),
        topology_digest="topo",
        bucket_contract_digest="dynamic_current",
    )


def test_raw_target_planner_does_not_build_paired_b() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore())
    _bundle, plan = service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="disabled"),
        metrics=TargetLayerPlannerMetrics(),
    )
    assert plan.selected_variant == "raw_u"
    assert plan.paired_b_logical_plan_digest == ""
    assert plan.safe_selection_us == 0.0
    assert plan.paired_b_build_us == 0.0


def test_safe_target_planner_builds_paired_b_and_records_selection() -> None:
    service = TargetLayerPlannerService(store=TargetPlanStore())
    _bundle, plan = service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="host_select"),
        metrics=TargetLayerPlannerMetrics(),
    )
    assert plan.raw_logical_plan_digest != ""
    assert plan.paired_b_logical_plan_digest != ""
    assert plan.selected_logical_plan_digest != ""
    assert plan.selected_variant in {"raw_u", "paired_b"}
    assert plan.paired_b_build_us >= 0.0
    assert plan.safe_selection_us >= 0.0
