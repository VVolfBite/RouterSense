"""Canonical multiphase flow-model contracts."""

from __future__ import annotations

from dataclasses import dataclass


EXECUTION_WINDOW_MODE = "execution_window"
RUNTIME_LOOKAHEAD_MODE = "runtime_lookahead"


@dataclass(frozen=True)
class ResidualFlowState:
    flow_id: str
    phase: int
    src_gpu: int
    dst_gpu: int
    volume: float
