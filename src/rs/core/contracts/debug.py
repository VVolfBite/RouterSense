from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Protocol


@dataclass(frozen=True)
class DebugEvent:
    event_type: str
    ts_ns: int
    performance_eligible: bool
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


class DebugProbe(Protocol):
    def record(self, event: DebugEvent) -> None:
        ...

    def flush(self) -> tuple[DebugEvent, ...]:
        ...
