"""Core logical flow contracts shared by offline analysis and scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FlowEdge:
    phase: str
    src_rank: int
    dst_rank: int
    row_count: int
    byte_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["FlowEdge"]
