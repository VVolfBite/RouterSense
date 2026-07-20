from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WaveDiagnostics:
    wave_id: int
    selected_flow_ids: tuple[str, ...]
    selected_edges: tuple[dict[str, Any], ...]
    matching_weight: float
    priority_components: dict[str, Any] = field(default_factory=dict)
    remaining_bytes_before: float = 0.0
    remaining_bytes_after: float = 0.0
    ready_flow_count_before: int = 0
    blocked_flow_count_before: int = 0
    forecast_pressure_summary: dict[str, Any] = field(default_factory=dict)
    selection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyDiagnostics:
    policy_name: str
    policy_version: str
    information_mode: str
    tie_break_rule: str
    wave_count: int
    logical_flow_count: int
    ready_flow_count: int
    blocked_flow_count: int
    forecast_flow_count: int
    p1_dependency_used: bool
    p2_forecast_used: bool
    p2_source: str
    evaluation_eligible: bool
    per_wave: tuple[WaveDiagnostics, ...] = ()
    priority_components: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
