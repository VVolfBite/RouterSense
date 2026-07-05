"""Algorithm-layer scheduling contracts shared by offline and online paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FlowDemand:
    flow_id: str
    phase: str
    src_rank: int
    dst_rank: int
    byte_count: int
    release_state: str
    is_executable: bool


@dataclass(frozen=True)
class FlowWindow:
    ready_flows: tuple[FlowDemand, ...] = ()
    blocked_flows: tuple[FlowDemand, ...] = ()
    forecast_pressure: tuple[FlowDemand, ...] = ()


@dataclass(frozen=True)
class LogicalWave:
    wave_id: int
    flows: tuple[FlowDemand, ...] = ()


@dataclass(frozen=True)
class LogicalSchedulePlan:
    policy_name: str
    waves: tuple[LogicalWave, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
