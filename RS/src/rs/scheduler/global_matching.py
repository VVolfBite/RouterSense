"""Compatibility re-export for the formal multiphase global ready-set scheduler."""

from __future__ import annotations

from rs.scheduling.multiphase.global_ready_set_impl import *  # noqa: F401,F403
from rs.scheduling.multiphase.global_ready_set_impl import (
    _collect_real_flows,
    _greedy_maximal_matching,
    _inbound_remaining,
    _maximum_weight_matching,
    _outbound_loads,
    _ready_flow_candidates,
    _run_global_matching_scheduler,
)

__all__ = [name for name in globals() if not name.startswith("__")]
