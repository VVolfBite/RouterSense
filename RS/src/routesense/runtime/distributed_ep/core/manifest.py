from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DistributedManifest:
    ranks: list[int]
    hosts: list[str]
    gpu_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
