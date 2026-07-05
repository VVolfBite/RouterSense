from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass(frozen=True)
class P2HintRequest:
    plan_key: dict[str, Any]
    layer_id: str
    phase: str
    global_rank: int
    local_rank: int
    ep_group_ranks: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class P2HintMetadata:
    hint_mode: str
    hint_digest: str
    hint_source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
