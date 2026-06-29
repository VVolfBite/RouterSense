from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..core.manifest import DispatchPlan, DispatchShard, RouteItem
from .expert_store import LocalExpertWeights


@dataclass
class OLMoEAdapterConfig:
    model_class: str
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_olmoe_adapter_config(model: Any) -> OLMoEAdapterConfig:
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("model has no config")
    return OLMoEAdapterConfig(
        model_class=model.__class__.__name__,
        num_experts=int(getattr(config, "num_experts")),
        top_k=int(getattr(config, "num_experts_per_tok")),
        hidden_size=int(getattr(config, "hidden_size")),
        intermediate_size=int(getattr(config, "intermediate_size")),
    )


def build_dispatch_plan_from_trace(
    trace_records: list[dict[str, Any]],
    *,
    owner_by_expert: dict[int, int],
    origin_rank: int,
    layer_id: int,
    world_size: int,
    request_id: str = "request-0",
    generation_step: int = 0,
) -> DispatchPlan:
    shard_map: dict[tuple[int, int], DispatchShard] = {}
    for record in trace_records:
        expert_id = int(record["expert_id"])
        destination_rank = int(owner_by_expert[expert_id])
        route_item = RouteItem(
            request_id=request_id,
            generation_step=generation_step,
            layer_id=layer_id,
            token_flat_index=int(record["token_position"]),
            route_rank_within_topk=int(record["expert_rank_within_topk"]),
            origin_rank=origin_rank,
            destination_rank=destination_rank,
            expert_id=expert_id,
            payload_rows=1,
            routing_weight=float(record["routing_weight"]),
            is_cross_node=origin_rank != destination_rank,
        )
        key = (origin_rank, destination_rank)
        shard = shard_map.setdefault(
            key,
            DispatchShard(source_rank=origin_rank, destination_rank=destination_rank),
        )
        shard.route_items.append(route_item)
    return DispatchPlan(layer_id=layer_id, world_size=world_size, shards=list(shard_map.values()))


def execute_local_experts(
    hidden_states: torch.Tensor,
    route_items: list[RouteItem],
    local_weights: LocalExpertWeights,
) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be rank-2, got {tuple(hidden_states.shape)}")
    if hidden_states.shape[0] != len(route_items):
        raise ValueError("hidden_states rows must match route_items length")
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, local_weights.hidden_dim))

    output = torch.zeros(
        (hidden_states.shape[0], local_weights.hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    local_index_by_expert = {expert_id: index for index, expert_id in enumerate(local_weights.local_expert_ids)}
    for row_index, route_item in enumerate(route_items):
        local_index = local_index_by_expert.get(route_item.expert_id)
        if local_index is None:
            raise RuntimeError(f"expert {route_item.expert_id} is not resident on this rank")
        current_state = hidden_states[row_index : row_index + 1]
        gate_up = local_weights.gate_up_proj[local_index]
        down = local_weights.down_proj[local_index]
        gate_up_value = F.linear(current_state, gate_up)
        gate, up = gate_up_value.chunk(2, dim=-1)
        current_hidden = F.silu(gate) * up
        current_hidden = F.linear(current_hidden, down)
        current_hidden = current_hidden * route_item.routing_weight
        output[row_index] = current_hidden[0].to(output.dtype)
    return output
