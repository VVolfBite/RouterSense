from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rs.core.contracts import PlanWave, PlannedFlow, PlanningRequest, WindowPlan
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave


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
    p3_return_weight: float = 0.0
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
        if family in {"phase_local_baseline", "deployable_baseline"}:
            return "baseline"
        return "local"
    if "joint" in scheduling_scope:
        if deployable:
            return "joint"
        return "exact_joint" if supports_p2_hint else "exact_local"
    return str(family)


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
    metadata.setdefault("policy_name", str(logical_plan.policy_name))
    metadata.setdefault("communication_makespan", float(metadata.get("makespan", sum(float(item.duration) for item in logical_plan.waves)) or 0.0))
    return WindowPlan(
        planner_id=str(planner_id),
        planner_family=str(planner_family),
        request_digest=request.semantic_digest(),
        waves=tuple(waves),
        metadata=metadata,
    )


def to_logical_plan(plan: WindowPlan) -> LogicalSchedulePlan:
    return LogicalSchedulePlan(
        policy_name=str(plan.metadata.get("policy_name", plan.planner_id)),
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


__all__ = [
    "Planner",
    "PlannerPolicyConfig",
    "PlannerSpec",
    "from_logical_plan",
    "planner_family_for_spec",
    "to_logical_plan",
]
