"""Project logical plans into current host-feasible runtime constraints.

Current Megatron host semantics only allow rank-level phase completion:

- token_dispatch returns after all local P0 roles complete
- local expert compute starts only after full local P0 completion
- token_combine proceeds only after full local P1 completion

This projector removes unsupported fine-grained overlap assumptions when
estimating runtime-safe Joint-candidate vs Local-fallback behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rs.scheduling.contracts import LogicalSchedulePlan


@dataclass(frozen=True)
class HostProjectedPlan:
    policy_name: str
    ideal_estimated_makespan: float
    host_projected_estimated_makespan: float
    projection_constraints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RUNTIME_HOST_PROJECTION_CONSTRAINTS = (
    "p1_release_requires_full_local_p0_completion",
    "p1_materialization_occurs_before_token_combine_only",
    "no_per_expert_compute_complete_event",
    "no_per_bucket_compute_overlap",
    "p2_is_advisory_only",
    "canonical_edge_order_preserved_within_phase",
)


class RuntimeHostFeasibilityProjector:
    def project(self, plan: LogicalSchedulePlan) -> HostProjectedPlan:
        ideal = float(plan.diagnostics.get("makespan", 0.0))
        raw_schedule = list(plan.diagnostics.get("raw_schedule", ()))
        if not raw_schedule:
            return HostProjectedPlan(
                policy_name=str(plan.policy_name),
                ideal_estimated_makespan=ideal,
                host_projected_estimated_makespan=ideal,
                projection_constraints=RUNTIME_HOST_PROJECTION_CONSTRAINTS,
            )
        # Conservative AR0 projection: P0 and P1 complete at rank granularity.
        rank_phase_end: dict[tuple[int, int], float] = {}
        projected_end = 0.0
        for step in raw_schedule:
            phase = int(step.get("phase", -1))
            src = int(step.get("src_gpu", -1))
            dst = int(step.get("dst_gpu", -1))
            start = float(step.get("start", 0.0))
            end = float(step.get("end", 0.0))
            duration = max(0.0, end - start)
            if phase == 0:
                rank_phase_end[(0, dst)] = max(rank_phase_end.get((0, dst), 0.0), end)
            elif phase == 1:
                # P1 materialization occurs at the expert/source rank after full local P0 completion.
                release_time = rank_phase_end.get((0, src), 0.0)
                start = max(start, release_time)
                end = start + duration
                rank_phase_end[(1, dst)] = max(rank_phase_end.get((1, dst), 0.0), end)
            projected_end = max(projected_end, end)
        return HostProjectedPlan(
            policy_name=str(plan.policy_name),
            ideal_estimated_makespan=ideal,
            host_projected_estimated_makespan=float(projected_end),
            projection_constraints=RUNTIME_HOST_PROJECTION_CONSTRAINTS,
        )


def host_project_safe_selection(
    *,
    joint_plan: LogicalSchedulePlan,
    local_plan: LogicalSchedulePlan,
) -> dict[str, Any]:
    projector = RuntimeHostFeasibilityProjector()
    joint_projection = projector.project(joint_plan)
    local_projection = projector.project(local_plan)
    select_local = (
        float(joint_projection.host_projected_estimated_makespan)
        > float(local_projection.host_projected_estimated_makespan)
    )
    return {
        "ideal_joint_candidate_estimated_makespan": float(joint_projection.ideal_estimated_makespan),
        "host_projected_joint_candidate_estimated_makespan": float(joint_projection.host_projected_estimated_makespan),
        "ideal_local_fallback_estimated_makespan": float(local_projection.ideal_estimated_makespan),
        "host_projected_local_fallback_estimated_makespan": float(local_projection.host_projected_estimated_makespan),
        "host_projected_safe_selection": str(local_plan.policy_name if select_local else joint_plan.policy_name),
        "projection_constraints": list(RUNTIME_HOST_PROJECTION_CONSTRAINTS),
    }


__all__ = [
    "HostProjectedPlan",
    "RUNTIME_HOST_PROJECTION_CONSTRAINTS",
    "RuntimeHostFeasibilityProjector",
    "host_project_safe_selection",
]
