from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass
class ExpertResidency:
    local_expert_ids: list[int] = field(default_factory=list)
    local_parameter_count: int = 0
    non_owner_parameter_count: int = 0
    weight_residency_mode: str = "rank_local_expert_weight_cache_from_full_model"

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
        weight_residency_mode="rank_local_expert_weight_cache_from_full_model",
    )


def plan_local_expert_ids(owner_by_expert: dict[int, int], rank: int) -> list[int]:
    return [expert_id for expert_id, owner_rank in sorted(owner_by_expert.items()) if owner_rank == rank]


def count_local_expert_parameters(experts_module: Any, local_expert_ids: list[int]) -> int:
    if experts_module is None:
        return 0
    if hasattr(experts_module, "__getitem__") and not hasattr(experts_module, "gate_up_proj"):
        total = 0
        for expert_id in local_expert_ids:
            expert = experts_module[int(expert_id)]
            total += sum(int(parameter.numel()) for parameter in expert.parameters())
        return total
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
    if hasattr(experts_module, "__getitem__") and not hasattr(experts_module, "gate_up_proj"):
        return _extract_modulelist_local_expert_weights(experts_module, local_expert_ids)
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


def _extract_modulelist_local_expert_weights(experts_module: Any, local_expert_ids: list[int]) -> LocalExpertWeights:
    if not local_expert_ids:
        first_expert = experts_module[0]
        gate_proj = getattr(first_expert, "gate_proj")
        up_proj = getattr(first_expert, "up_proj")
        down_proj = getattr(first_expert, "down_proj")
        hidden_dim = int(gate_proj.weight.shape[-1])
        intermediate_dim = int(gate_proj.weight.shape[0])
        empty_gate_up = gate_proj.weight.new_empty((0, intermediate_dim * 2, hidden_dim))
        empty_down = down_proj.weight.new_empty((0, hidden_dim, intermediate_dim))
        return LocalExpertWeights(
            local_expert_ids=[],
            gate_up_proj=empty_gate_up,
            down_proj=empty_down,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation_name="silu",
        )

    gate_up_rows: list[torch.Tensor] = []
    down_rows: list[torch.Tensor] = []
    hidden_dim: int | None = None
    intermediate_dim: int | None = None
    for expert_id in local_expert_ids:
        expert = experts_module[int(expert_id)]
        gate_proj = getattr(expert, "gate_proj", None)
        up_proj = getattr(expert, "up_proj", None)
        down_proj = getattr(expert, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            raise RuntimeError("expert module does not expose gate_proj/up_proj/down_proj")
        gate_weight = gate_proj.weight.detach().clone()
        up_weight = up_proj.weight.detach().clone()
        down_weight = down_proj.weight.detach().clone()
        gate_up_rows.append(torch.cat([gate_weight, up_weight], dim=0))
        down_rows.append(down_weight)
        hidden_dim = int(gate_weight.shape[-1])
        intermediate_dim = int(gate_weight.shape[0])

    assert hidden_dim is not None
    assert intermediate_dim is not None
    return LocalExpertWeights(
        local_expert_ids=list(local_expert_ids),
        gate_up_proj=torch.stack(gate_up_rows, dim=0),
        down_proj=torch.stack(down_rows, dim=0),
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        activation_name="silu",
    )
