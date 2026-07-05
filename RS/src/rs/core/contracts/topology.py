"""Topology and placement contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TopologySnapshot:
    world_size: int
    node_count: int
    rank_to_host: dict[int, str]
    rank_to_gpu: dict[int, str]
    topology_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlacementSnapshot:
    model_revision: str | None
    transport_backend: str
    expert_placement: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["PlacementSnapshot", "TopologySnapshot"]
