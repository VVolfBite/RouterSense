from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyDiagnostics:
    policy_name: str
    policy_version: str
    policy_capabilities: dict[str, Any]
    phase: str
    layer_id: str
    plan_hash: str
    bucket_count: int
    wave_count: int
    bucket_order: list[str] = field(default_factory=list)
    wave_edges: list[list[dict[str, Any]]] = field(default_factory=list)
    per_wave_matching_weight: list[float] = field(default_factory=list)
    uses_current_phase_demand: bool = True
    uses_p1_reservation: bool = False
    uses_p2_hint: bool = False
    priority_components: dict[str, Any] = field(default_factory=dict)
    tie_break_rule: str = ""
    fallback_reason: str = ""
    evaluation_eligible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
