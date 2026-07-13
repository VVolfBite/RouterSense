from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rs.core.contracts import PlanWave, PlannedFlow, PlanningRequest, WindowPlan
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave
from rs.scheduling.unified_interface import SchedulingRequest, SchedulingTopology, build_policy as build_legacy_policy

from .legacy_aliases import normalize_family_name


class Planner(Protocol):
    @property
    def planner_id(self) -> str:
        ...

    @property
    def planner_family(self) -> str:
        ...

    def plan(self, request: PlanningRequest) -> WindowPlan:
        ...


@dataclass(frozen=True)
class PlannerSpec:
    planner_id: str
    planner_family: str
    deployable: bool
    reference_only: bool
    requires_prediction: bool
    exact: bool
    historical_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannerPolicyConfig:
    p0_weight: float = 1.0
    p1_weight: float = 1.0
    p2_hint_weight: float = 1.0
    residual_weight: float = 0.75
    barrier_weight: float = 1.75
    age_weight: float = 0.15
    prediction_weight: float = 0.35
    criticality_weight: float = 0.0
    iteration_budget: int | None = None


def planner_family_for_spec(*, family: str, scheduling_scope: str, reference_only: bool, deployable: bool, supports_p2_hint: bool, canonical_id: str, execution_model: str = "") -> str:
    if reference_only:
        if execution_model == "exact_reference" or family == "oracle":
            return "exact_local" if ("local" in canonical_id or "phase_local" in scheduling_scope) else "exact_joint"
        return "reference_local" if "local" in canonical_id or "phase_local" in scheduling_scope else "reference_joint"
    if scheduling_scope == "phase_local":
        if family.startswith("phase_local_baseline"):
            return "baseline"
        return "local"
    if "joint" in scheduling_scope:
        if deployable:
            return "joint"
        return "exact_joint" if supports_p2_hint else "exact_local"
    return normalize_family_name(family)


def to_legacy_request(request: PlanningRequest) -> SchedulingRequest:
    from rs.scheduling.unified_interface import PlanningHintMetadata, PolicyOptions

    request.validate()
    return SchedulingRequest(
        request_id=str(request.identity.request_id),
        tasks=(),
        p0_truth_rows=request.traffic.p0_dispatch_rows,
        p1_truth_rows=request.traffic.p1_return_rows,
        p2_hint_rows=request.prediction_hint.target_dispatch_rows,
        topology=SchedulingTopology(group_size=int(request.topology.world_size)),
        release_model=_release_constraint(request),
        policy_options=PolicyOptions(
            p0_weight=float(request.weights.p0_weight),
            p1_weight=float(request.weights.p1_weight),
            p2_hint_weight=float(request.weights.p2_weight),
            residual_weight=float(request.weights.residual_weight),
            barrier_weight=float(request.weights.barrier_weight),
            age_weight=float(request.weights.age_weight),
            prediction_weight=float(request.weights.prediction_weight),
            criticality_weight=float(request.weights.criticality_weight),
            iteration_budget=request.weights.iteration_budget,
        ),
        hint_metadata=PlanningHintMetadata(
            hint_type=str(request.prediction_hint.hint_type),
            confidence=float(request.prediction_hint.confidence or 0.0),
            source_layer=None if request.identity.source_layer_id is None else int(request.identity.source_layer_id) if str(request.identity.source_layer_id).isdigit() else None,
            target_layer=None if request.identity.target_layer_id is None else int(request.identity.target_layer_id) if str(request.identity.target_layer_id).isdigit() else None,
        ),
        scheduling_mode="execution_window",
        information_mode=str(request.information_mode),
        max_waves=int(request.constraints.max_waves),
        task_quantum_rows=int(request.constraints.bucket_rows),
        fixture_id=str(request.identity.run_id or "runtime"),
        window_id=str(request.identity.window_id or request.identity.request_id),
        layer_id=None if request.identity.source_layer_id is None or not str(request.identity.source_layer_id).isdigit() else int(request.identity.source_layer_id),
    )


def build_runtime_policy(policy_name: str, options: PlannerPolicyConfig):
    from rs.scheduling.unified_interface import PolicyOptions

    return build_legacy_policy(
        policy_name,
        PolicyOptions(
            p0_weight=float(options.p0_weight),
            p1_weight=float(options.p1_weight),
            p2_hint_weight=float(options.p2_hint_weight),
            residual_weight=float(options.residual_weight),
            barrier_weight=float(options.barrier_weight),
            age_weight=float(options.age_weight),
            prediction_weight=float(options.prediction_weight),
            criticality_weight=float(options.criticality_weight),
            iteration_budget=options.iteration_budget,
        ),
    )


def build_runtime_request_from_problem(
    *,
    request_id: str,
    problem,
    bucket_rows: int,
    policy_options: PlannerPolicyConfig,
    hint_type: str,
    confidence: float,
    layer_id: int | None,
):
    from rs.scheduling.unified_interface import PlanningHintMetadata, PolicyOptions, SchedulingRequest, SchedulingTopology

    return SchedulingRequest(
        request_id=str(request_id),
        tasks=(),
        p0_truth_rows=problem.p0_dispatch_matrix,
        p1_truth_rows=problem.p1_return_matrix,
        p2_hint_rows=problem.p2_next_dispatch_forecast_matrix,
        topology=SchedulingTopology(group_size=int(problem.topology.num_gpus)),
        release_model=problem.release_model,
        policy_options=PolicyOptions(
            p0_weight=float(policy_options.p0_weight),
            p1_weight=float(policy_options.p1_weight),
            p2_hint_weight=float(policy_options.p2_hint_weight),
            residual_weight=float(policy_options.residual_weight),
            barrier_weight=float(policy_options.barrier_weight),
            age_weight=float(policy_options.age_weight),
            prediction_weight=float(policy_options.prediction_weight),
            criticality_weight=float(policy_options.criticality_weight),
            iteration_budget=policy_options.iteration_budget,
        ),
        hint_metadata=PlanningHintMetadata(
            hint_type=str(hint_type),
            confidence=float(confidence),
            source_layer=int(layer_id) if layer_id is not None else None,
            target_layer=(int(layer_id) + 1) if layer_id is not None else None,
        ),
        scheduling_mode=str(problem.options.scheduling_mode),
        information_mode=str(problem.options.information_mode),
        max_waves=int(problem.options.max_waves),
        task_quantum_rows=int(bucket_rows),
        fixture_id=str(getattr(problem.forecast, "metadata", {}).get("fixture_id", "runtime")),
        window_id=str(getattr(problem.forecast, "metadata", {}).get("replay_window_id", request_id)),
        layer_id=None if layer_id is None else int(layer_id),
    )


def _release_constraint(request: PlanningRequest):
    from rs.scheduling.contracts import ReleaseConstraint

    return ReleaseConstraint(
        phase=str(request.constraints.phase_release_model),
        rank=0,
        release_after_phase="p0_dispatch",
        expert_compute_delay=float(request.constraints.expert_compute_delay),
    )


def from_logical_plan(
    *,
    planner_id: str,
    planner_family: str,
    request: PlanningRequest,
    logical_plan: LogicalSchedulePlan,
) -> WindowPlan:
    waves: list[PlanWave] = []
    for wave in logical_plan.waves:
        flows = tuple(
            PlannedFlow(
                flow_id=str(flow.flow_id),
                phase=str(flow.phase),
                src_rank=int(flow.src_rank),
                dst_rank=int(flow.dst_rank),
                row_count=int(flow.byte_count),
                release_state=str(flow.release_state),
                executable=bool(flow.is_executable),
            )
            for flow in wave.flows
        )
        waves.append(
            PlanWave(
                wave_id=int(wave.wave_id),
                flows=flows,
                estimated_duration=float(wave.duration),
            )
        )
    metadata = dict(getattr(logical_plan, "diagnostics", {}) or {})
    metadata.setdefault("legacy_policy_name", str(logical_plan.policy_name))
    metadata.setdefault("legacy_makespan", float(metadata.get("makespan", sum(float(item.duration) for item in logical_plan.waves)) or 0.0))
    return WindowPlan(
        planner_id=str(planner_id),
        planner_family=str(planner_family),
        request_digest=request.semantic_digest(),
        waves=tuple(waves),
        metadata=metadata,
    )


def to_logical_plan(plan: WindowPlan) -> LogicalSchedulePlan:
    return LogicalSchedulePlan(
        policy_name=str(plan.metadata.get("legacy_policy_name", plan.planner_id)),
        waves=tuple(
            LogicalWave(
                wave_id=int(wave.wave_id),
                flows=tuple(
                    FlowDemand(
                        flow_id=str(flow.flow_id),
                        phase=str(flow.phase),
                        src_rank=int(flow.src_rank),
                        dst_rank=int(flow.dst_rank),
                        byte_count=int(flow.row_count),
                        release_state=str(flow.release_state),
                        is_executable=bool(flow.executable),
                    )
                    for flow in wave.flows
                ),
                duration=float(wave.estimated_duration),
            )
            for wave in plan.waves
        ),
        diagnostics=dict(plan.metadata),
    )


@dataclass
class LegacyPlannerAdapter(Planner):
    _planner_id: str
    _planner_family: str
    _builder_key: str

    @property
    def planner_id(self) -> str:
        return self._planner_id

    @property
    def planner_family(self) -> str:
        return self._planner_family

    def plan(self, request: PlanningRequest) -> WindowPlan:
        legacy_request = to_legacy_request(request)
        policy = build_legacy_policy(self._planner_id, legacy_request.policy_options)
        logical_plan = policy.plan(legacy_request)
        return from_logical_plan(
            planner_id=self._planner_id,
            planner_family=self._planner_family,
            request=request,
            logical_plan=logical_plan,
        )


__all__ = [
    "LegacyPlannerAdapter",
    "Planner",
    "PlannerPolicyConfig",
    "PlannerSpec",
    "from_logical_plan",
    "planner_family_for_spec",
    "to_logical_plan",
    "to_legacy_request",
]
