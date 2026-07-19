"""Observed online-EP trace contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any


def stable_hash_dict(payload: dict[str, Any]) -> str:
    import json

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OnlineRouteIdentity:
    run_id: str
    request_id: str
    microbatch_id: str
    layer_id: int
    source_rank: int
    destination_rank: int
    source_node_id: int
    destination_node_id: int
    local_token_index: int
    topk_slot: int
    expert_id: int
    request_numeric_id: int = 0
    microbatch_numeric_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineRouteRecord:
    identity: OnlineRouteIdentity
    routing_weight: float
    payload_rows: int
    payload_bytes: int
    is_local_route: bool
    is_remote_route: bool
    is_cross_rank: bool
    is_cross_node: bool

    def __post_init__(self) -> None:
        is_local = self.identity.source_rank == self.identity.destination_rank
        is_remote = self.identity.source_rank != self.identity.destination_rank
        is_cross_rank = is_remote
        is_cross_node = self.identity.source_node_id != self.identity.destination_node_id
        if self.is_local_route != is_local:
            raise ValueError("is_local_route must match source_rank == destination_rank")
        if self.is_remote_route != is_remote:
            raise ValueError("is_remote_route must match source_rank != destination_rank")
        if self.is_cross_rank != is_cross_rank:
            raise ValueError("is_cross_rank must match source_rank != destination_rank")
        if self.is_cross_node != is_cross_node:
            raise ValueError("is_cross_node must match source_node_id != destination_node_id")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["identity"] = self.identity.to_dict()
        return payload


@dataclass(frozen=True)
class OnlineLayerRouteTrace:
    trace_origin: str
    future_information_mode: str
    layer_id: int
    local_routes: list[OnlineRouteRecord] = field(default_factory=list)
    remote_send_routes: list[OnlineRouteRecord] = field(default_factory=list)
    all_routes: list[OnlineRouteRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_origin": self.trace_origin,
            "future_information_mode": self.future_information_mode,
            "layer_id": self.layer_id,
            "local_routes": [record.to_dict() for record in self.local_routes],
            "remote_send_routes": [record.to_dict() for record in self.remote_send_routes],
            "all_routes": [record.to_dict() for record in self.all_routes],
        }


@dataclass(frozen=True)
class OnlineExpertPlacement:
    world_size: int
    expert_count: int
    owner_rank_by_expert: list[int]
    owner_node_id_by_expert: list[int]
    placement_hash: str
    placement_mode: str = "expert_id_mod_world_size"
    residency_mode: str = "full_checkpoint_then_local_extract"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankManifest:
    run_id: str
    request_id: str
    microbatch_id: str
    request_numeric_id: int
    microbatch_numeric_id: int
    layer_id: int
    rank: int
    world_size: int
    node_id: int
    placement_hash: str
    request_protocol_hash: str
    request_table_hash: str
    prompt_digest: str
    route_count: int
    local_route_count: int
    remote_route_count: int
    remote_send_row_count: int
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransportOperationRecord:
    run_id: str
    rank: int
    local_rank: int | None
    world_size: int
    operation_id: str
    phase: str
    backend: str
    verified_backend: str | None
    device: str | None
    operation_kind: str
    hidden_payload_transferred: bool
    send_counts: list[int]
    recv_counts: list[int]
    send_rows: int
    recv_rows: int
    send_bytes: int
    recv_bytes: int
    wall_elapsed_ms: float
    success: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "OnlineExpertPlacement",
    "OnlineLayerRouteTrace",
    "OnlineRouteIdentity",
    "OnlineRouteRecord",
    "RankManifest",
    "TransportOperationRecord",
    "stable_hash_dict",
]
