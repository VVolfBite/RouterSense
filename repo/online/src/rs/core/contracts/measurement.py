from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Protocol


@dataclass(frozen=True)
class MeasurementEvent:
    event_type: str
    started_at_ns: int
    ended_at_ns: int
    layer_id: str | None = None
    phase: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.started_at_ns) < 0 or int(self.ended_at_ns) < int(self.started_at_ns):
            raise ValueError("measurement event timestamps must be non-negative and ordered")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class MeasurementSnapshot:
    event_count: int
    dropped_event_count: int
    instrumentation_mode: str
    events: tuple[MeasurementEvent, ...] = ()
    summary: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_count": int(self.event_count),
            "dropped_event_count": int(self.dropped_event_count),
            "instrumentation_mode": str(self.instrumentation_mode),
            "events": [event.to_dict() for event in self.events],
            "summary": dict(self.summary),
        }


class MeasurementSink(Protocol):
    def record(self, event: MeasurementEvent) -> None:
        ...

    def snapshot(self) -> MeasurementSnapshot:
        ...

    def reset(self) -> None:
        ...
