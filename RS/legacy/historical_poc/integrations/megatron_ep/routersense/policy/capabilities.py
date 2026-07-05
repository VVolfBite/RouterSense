from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyCapabilities:
    uses_p0: bool
    uses_p1: bool
    uses_p2: bool
    cross_phase: bool
    requires_topology: bool
    supports_sync_before_phase: bool
    supports_default_continue: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
