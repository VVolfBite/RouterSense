"""Formal scheduling package for RouterSense.

Keep package import light. Most subpackages are resolved lazily so callers can
import a narrow helper module without eagerly importing the full registry and
policy stack.
"""

from __future__ import annotations

__all__ = [
    "FlowDemand",
    "ForecastPressure",
    "FlowWindow",
    "GlobalReadySetOptions",
    "LogicalSchedulePlan",
    "LogicalTopology",
    "LogicalWave",
    "MultiPhaseSchedulingProblem",
    "PreparedWindowPlan",
    "ReleaseConstraint",
    "estimate_planning_quantum_rows_from_contexts",
    "estimate_planning_quantum_rows_from_values",
    "family_inventory",
    "resolve_phase_policy",
    "resolve_policy",
    "supported_phase_policies",
    "supported_policies",
]


def __getattr__(name: str):
    if name in {
        "FlowDemand",
        "FlowWindow",
        "ForecastPressure",
        "GlobalReadySetOptions",
        "LogicalSchedulePlan",
        "LogicalTopology",
        "LogicalWave",
        "MultiPhaseSchedulingProblem",
        "PreparedWindowPlan",
        "ReleaseConstraint",
    }:
        from .contracts import (
            FlowDemand,
            FlowWindow,
            ForecastPressure,
            GlobalReadySetOptions,
            LogicalSchedulePlan,
            LogicalTopology,
            LogicalWave,
            MultiPhaseSchedulingProblem,
            PreparedWindowPlan,
            ReleaseConstraint,
        )

        return {
            "FlowDemand": FlowDemand,
            "FlowWindow": FlowWindow,
            "ForecastPressure": ForecastPressure,
            "GlobalReadySetOptions": GlobalReadySetOptions,
            "LogicalSchedulePlan": LogicalSchedulePlan,
            "LogicalTopology": LogicalTopology,
            "LogicalWave": LogicalWave,
            "MultiPhaseSchedulingProblem": MultiPhaseSchedulingProblem,
            "PreparedWindowPlan": PreparedWindowPlan,
            "ReleaseConstraint": ReleaseConstraint,
        }[name]
    if name == "family_inventory":
        from .families import family_inventory

        return family_inventory
    if name in {"resolve_phase_policy", "resolve_policy", "supported_phase_policies", "supported_policies"}:
        from .registry import resolve_phase_policy, resolve_policy, supported_phase_policies, supported_policies

        return {
            "resolve_phase_policy": resolve_phase_policy,
            "resolve_policy": resolve_policy,
            "supported_phase_policies": supported_phase_policies,
            "supported_policies": supported_policies,
        }[name]
    if name in {"estimate_planning_quantum_rows_from_contexts", "estimate_planning_quantum_rows_from_values"}:
        from .phase_local.common import estimate_planning_quantum_rows_from_contexts, estimate_planning_quantum_rows_from_values

        return {
            "estimate_planning_quantum_rows_from_contexts": estimate_planning_quantum_rows_from_contexts,
            "estimate_planning_quantum_rows_from_values": estimate_planning_quantum_rows_from_values,
        }[name]
    raise AttributeError(name)
