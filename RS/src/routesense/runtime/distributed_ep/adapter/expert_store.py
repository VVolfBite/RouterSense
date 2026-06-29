from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExpertResidency:
    local_expert_ids: list[int] = field(default_factory=list)
    local_parameter_count: int = 0
    non_owner_parameter_count: int = 0
    weight_residency_mode: str = "physically_sharded_experts"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_residency(local_expert_ids: list[int], *, local_parameter_count: int) -> ExpertResidency:
    return ExpertResidency(
        local_expert_ids=sorted(local_expert_ids),
        local_parameter_count=local_parameter_count,
        non_owner_parameter_count=0,
        weight_residency_mode="physically_sharded_experts",
    )


def plan_local_expert_ids(owner_by_expert: dict[int, int], rank: int) -> list[int]:
    return [expert_id for expert_id, owner_rank in sorted(owner_by_expert.items()) if owner_rank == rank]


def count_local_expert_parameters(experts_module: Any, local_expert_ids: list[int]) -> int:
    if experts_module is None:
        return 0
    total = 0
    names = list(experts_module.named_parameters()) if hasattr(experts_module, "named_parameters") else []
    if not names:
        return 0
    expert_tokens = {str(expert_id) for expert_id in local_expert_ids}
    for name, parameter in names:
        if any(token in name for token in expert_tokens):
            total += int(parameter.numel())
    return total
