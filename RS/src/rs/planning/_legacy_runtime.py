from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import PlanningRequest
from rs.scheduling.contracts import ReleaseConstraint
from rs.scheduling.unified_interface import SchedulingRequest, SchedulingTopology, build_policy as build_legacy_policy

from .api import Planner, PlannerPolicyConfig, PlannerSpec, from_logical_plan


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
        release_model=ReleaseConstraint(
            phase=str(request.constraints.phase_release_model),
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=float(request.constraints.expert_compute_delay),
        ),
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

    def plan(self, request: PlanningRequest):
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
    "PlannerPolicyConfig",
    "build_runtime_policy",
    "build_runtime_request_from_problem",
    "to_legacy_request",
]
