from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RouteItem:
    request_id: str
    generation_step: int
    layer_id: int
    token_flat_index: int
    route_rank_within_topk: int
    origin_rank: int
    destination_rank: int
    expert_id: int
    payload_rows: int
    routing_weight: float
    is_cross_node: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DispatchShard:
    source_rank: int
    destination_rank: int
    route_items: list[RouteItem] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return sum(item.payload_rows for item in self.route_items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_rank": self.source_rank,
            "destination_rank": self.destination_rank,
            "rows": self.rows,
            "route_items": [item.to_dict() for item in self.route_items],
        }


@dataclass
class DispatchPlan:
    layer_id: int
    world_size: int
    shards: list[DispatchShard]

    def send_counts_for_rank(self, rank: int) -> list[int]:
        counts = [0 for _ in range(self.world_size)]
        for shard in self.shards:
            if shard.source_rank == rank:
                counts[shard.destination_rank] += shard.rows
        return counts

    def recv_counts_for_rank(self, rank: int) -> list[int]:
        counts = [0 for _ in range(self.world_size)]
        for shard in self.shards:
            if shard.destination_rank == rank:
                counts[shard.source_rank] += shard.rows
        return counts

    def total_cross_node_routes(self) -> int:
        return sum(
            len(shard.route_items)
            for shard in self.shards
            if shard.source_rank != shard.destination_rank
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "world_size": self.world_size,
            "shards": [shard.to_dict() for shard in self.shards],
        }


@dataclass
class DistributedManifest:
    ranks: list[int]
    hosts: list[str]
    gpu_names: list[str] = field(default_factory=list)
    placement: dict[int, int] = field(default_factory=dict)
    dispatch_plans: list[DispatchPlan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranks": self.ranks,
            "hosts": self.hosts,
            "gpu_names": self.gpu_names,
            "placement": self.placement,
            "dispatch_plans": [plan.to_dict() for plan in self.dispatch_plans],
            "metadata": self.metadata,
        }
