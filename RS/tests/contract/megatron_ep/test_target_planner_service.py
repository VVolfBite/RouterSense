from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import PlanWave, PlannedFlow, PlanningRequest, WindowPlan
from rs.planning import PlannerPolicyConfig, PlannerSelectionMode
from rs.runtime.online.megatron_ep.target_planning.planner_service import (
    TargetLayerPlannerMetrics,
    TargetLayerPlannerService,
    TargetLayerPlanningRequest,
)
from rs.runtime.online.megatron_ep.target_planning.store import TargetPlanStore


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
        policy_options=PlannerPolicyConfig(),
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


@dataclass
class _CountingPlanner:
    planner_id: str
    planner_family: str
    counter: dict[str, int]

    def plan(self, request: PlanningRequest) -> WindowPlan:
        self.counter[self.planner_id] = int(self.counter.get(self.planner_id, 0)) + 1
        return WindowPlan(
            planner_id=self.planner_id,
            planner_family=self.planner_family,
            request_digest=request.semantic_digest(),
            waves=(
                PlanWave(
                    wave_id=0,
                    flows=(PlannedFlow(flow_id=f"{self.planner_id}:0", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=2, release_state="ready", executable=True),),
                    estimated_duration=0.0,
                ),
            ),
            metadata={"legacy_policy_name": self.planner_id},
        )


def test_target_planner_service_core_selection_does_not_replan() -> None:
    counter: dict[str, int] = {}

    def planner_factory(planner_id: str, _config) -> _CountingPlanner:
        family = "joint" if "joint" in planner_id or planner_id.startswith("U_") else "local"
        return _CountingPlanner(planner_id=planner_id, planner_family=family, counter=counter)

    service = TargetLayerPlannerService(store=TargetPlanStore(), planner_factory=planner_factory)
    from rs.core.contracts import (
        PlanningConstraints,
        PlanningIdentity,
        PlanningRequest,
        PlanningTopology,
        PlanningTraffic,
        PlanningWeights,
        PredictionHint,
    )

    formal_request = PlanningRequest(
        identity=PlanningIdentity(request_id="req"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 2), (2, 0)), p1_return_rows=((0, 2), (2, 0))),
        prediction_hint=PredictionHint(predictor_id="copy_current", hint_type="traffic_matrix", target_dispatch_rows=((0, 2), (2, 0)), confidence=1.0),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=2, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    local_planner = planner_factory("fifo_bucket", None)
    joint_planner = planner_factory("barrier_criticality_joint", None)
    local_plan = local_planner.plan(formal_request)
    compare_selector = service._select_candidate_plans(  # noqa: SLF001
        planning_request=formal_request,
        local_plan=local_plan,
        joint_plan=None,
        mode=PlannerSelectionMode.LOCAL,
    )
    assert compare_selector.selected_plan.planner_id == "fifo_bucket"
    assert counter["fifo_bucket"] == 1
    assert counter.get("barrier_criticality_joint", 0) == 0
    counter.clear()
    joint_plan = joint_planner.plan(formal_request)
    compare_selector = service._select_candidate_plans(  # noqa: SLF001
        planning_request=formal_request,
        local_plan=None,
        joint_plan=joint_plan,
        mode=PlannerSelectionMode.JOINT,
    )
    assert compare_selector.selected_plan.planner_id == "barrier_criticality_joint"
    assert counter.get("fifo_bucket", 0) == 0
    assert counter["barrier_criticality_joint"] == 1
    counter.clear()
    local_plan = local_planner.plan(formal_request)
    joint_plan = joint_planner.plan(formal_request)
    compare_selector = service._select_candidate_plans(  # noqa: SLF001
        planning_request=formal_request,
        local_plan=local_plan,
        joint_plan=joint_plan,
        mode=PlannerSelectionMode.COMPARE,
    )
    assert counter["fifo_bucket"] == 1
    assert counter["barrier_criticality_joint"] == 1


def test_target_planner_build_path_counts_planner_calls_once_per_mode() -> None:
    counter: dict[str, int] = {}

    def planner_factory(planner_id: str, _config) -> _CountingPlanner:
        family = "joint" if "joint" in planner_id or planner_id.startswith("U_") else "local"
        return _CountingPlanner(planner_id=planner_id, planner_family=family, counter=counter)

    service = TargetLayerPlannerService(store=TargetPlanStore(), planner_factory=planner_factory)
    service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="disabled"),
        metrics=TargetLayerPlannerMetrics(),
    )
    assert counter["U_barrier_criticality_global_matching"] == 1
    assert counter.get("B_barrier_criticality_core_independent", 0) == 0
    counter.clear()
    service._build_target_plan(  # noqa: SLF001
        request=_request(safe_projection_mode="host_select"),
        metrics=TargetLayerPlannerMetrics(),
    )
    assert counter["U_barrier_criticality_global_matching"] == 1
    assert counter["B_barrier_criticality_core_independent"] == 1
