"""Pending-window 子包：负责在线联合规划的窗口状态、shadow 构造与 phase plan 编译。"""

from .adapter import MultiphasePendingWindowAdapter, UnsupportedPendingWindowAdapter
from .policy_adapter import compile_prepared_window_phase_plan
from .release_engine import record_release_event
from .shadow import build_pending_window_shadow, classify_flow, executable_now
from .window_state import PreparedPlanBinding, WindowReleaseState, bind_prepared_plan, build_shadow_problem, build_window_state

__all__ = [
    "MultiphasePendingWindowAdapter",
    "PreparedPlanBinding",
    "UnsupportedPendingWindowAdapter",
    "WindowReleaseState",
    "bind_prepared_plan",
    "build_pending_window_shadow",
    "build_shadow_problem",
    "build_window_state",
    "classify_flow",
    "compile_prepared_window_phase_plan",
    "executable_now",
    "record_release_event",
]
