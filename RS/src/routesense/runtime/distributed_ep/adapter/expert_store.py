from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExpertResidency:
    local_expert_ids: list[int] = field(default_factory=list)
    local_parameter_count: int = 0
    non_owner_parameter_count: int = 0
    weight_residency_mode: str = "physically_sharded_experts"
