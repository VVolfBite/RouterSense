"""Private migrated P012/Future algorithm kernel.

This package is intentionally private.  Formal online/offline callers must use
``rs.planning.p012_future`` or ``rs.prediction.fate_future`` and must never
publish the kernel's standalone contracts as RouterSense authority.
"""

from .axes import PlannerAxes, parse_planner_axes, planner_axis_matrix
from .families import (
    EventJointPlanner,
    EventUPlanner,
    GlobalJointPlanner,
    GlobalOrderingUPlanner,
    LocalBPlanner,
    LocalScopedPlanner,
    build_scoped_planner,
)
from .future import FutureP012Planner
from .registry import build_planner, planner_matrix

__all__ = [
    "EventJointPlanner",
    "EventUPlanner",
    "FutureP012Planner",
    "GlobalJointPlanner",
    "GlobalOrderingUPlanner",
    "LocalBPlanner",
    "LocalScopedPlanner",
    "PlannerAxes",
    "build_scoped_planner",
    "parse_planner_axes",
    "planner_axis_matrix",
    "build_planner",
    "planner_matrix",
]
