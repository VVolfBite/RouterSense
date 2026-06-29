from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..core.manifest import DispatchPlan, DispatchShard, RouteItem


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
