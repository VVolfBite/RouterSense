"""Formal scheduling package for RouterSense."""

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
from .registry import resolve_phase_policy, resolve_policy, supported_phase_policies, supported_policies
from .phase_local.common import estimate_planning_quantum_rows_from_contexts, estimate_planning_quantum_rows_from_values

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
    "resolve_phase_policy",
    "resolve_policy",
    "supported_phase_policies",
    "supported_policies",
]
