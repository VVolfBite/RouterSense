from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteIdentity:
    run_id: str
    request_id: str
    microbatch_id: str
    layer_id: int
    source_rank: int
    destination_rank: int
    expert_id: int
    token_index_local: int
    topk_slot: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteRecord:
    identity: RouteIdentity
    routing_weight: float
    payload_rows: int
    payload_bytes: int
    is_local_route: bool
    is_remote_route: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["identity"] = self.identity.to_dict()
        return payload


@dataclass
class LayerRouteTrace:
    trace_origin: str
    future_information_mode: str
    layer_id: int
    route_records: list[RouteRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_origin": self.trace_origin,
            "future_information_mode": self.future_information_mode,
            "layer_id": self.layer_id,
            "route_records": [record.to_dict() for record in self.route_records],
        }


@dataclass(frozen=True)
class ExpertBucketRecord:
    rank: int
    layer_id: int
    expert_id: int
    bucket_rows: int
    bucket_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
