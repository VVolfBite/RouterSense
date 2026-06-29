from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass
class ExpertResidency:
    local_expert_ids: list[int] = field(default_factory=list)
    local_parameter_count: int = 0
    non_owner_parameter_count: int = 0
    weight_residency_mode: str = "physically_sharded_experts"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalExpertWeights:
    local_expert_ids: list[int]
    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor
    hidden_dim: int
    intermediate_dim: int
    activation_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_expert_ids": self.local_expert_ids,
            "gate_up_proj_shape": list(self.gate_up_proj.shape),
            "down_proj_shape": list(self.down_proj.shape),
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "activation_name": self.activation_name,
        }


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


def extract_local_expert_weights(experts_module: Any, local_expert_ids: list[int]) -> LocalExpertWeights:
    if experts_module is None:
        raise ValueError("experts_module must not be None")
    gate_up_proj = getattr(experts_module, "gate_up_proj", None)
    down_proj = getattr(experts_module, "down_proj", None)
    if gate_up_proj is None or down_proj is None:
        raise RuntimeError("experts_module does not expose gate_up_proj/down_proj")
    if not local_expert_ids:
        hidden_dim = int(gate_up_proj.shape[-1])
        intermediate_dim = int(gate_up_proj.shape[1] // 2)
        return LocalExpertWeights(
            local_expert_ids=[],
            gate_up_proj=gate_up_proj.new_empty((0, gate_up_proj.shape[1], gate_up_proj.shape[2])),
            down_proj=down_proj.new_empty((0, down_proj.shape[1], down_proj.shape[2])),
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation_name=getattr(getattr(experts_module, "act_fn", None), "__name__", "unknown"),
        )
    index = torch.tensor(local_expert_ids, device=gate_up_proj.device, dtype=torch.long)
    local_gate_up = gate_up_proj.index_select(0, index).detach().clone()
    local_down = down_proj.index_select(0, index).detach().clone()
    return LocalExpertWeights(
        local_expert_ids=list(local_expert_ids),
        gate_up_proj=local_gate_up,
        down_proj=local_down,
        hidden_dim=int(local_gate_up.shape[-1]),
        intermediate_dim=int(local_gate_up.shape[1] // 2),
        activation_name=getattr(getattr(experts_module, "act_fn", None), "__name__", "unknown"),
    )
