from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rs.core.contracts import PlanWave, PlannedFlow, PlanningRequest, WindowPlan
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave
from rs.scheduling.unified_interface import (
    PolicyOptions,
    SchedulingRequest,
    SchedulingTopology,
    build_policy as build_legacy_policy,
)

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


def planner_family_for_spec(*, family: str, scheduling_scope: str, reference_only: bool, deployable: bool, supports_p2_hint: bool, canonical_id: str) -> str:
    if reference_only:
        if "local" in canonical_id or "phase_local" in scheduling_scope:
            return "exact_local"
        return "exact_joint"
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
    from rs.scheduling.unified_interface import PlanningHintMetadata

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
    "PlannerSpec",
    "from_logical_plan",
    "planner_family_for_spec",
    "to_logical_plan",
    "to_legacy_request",
]
