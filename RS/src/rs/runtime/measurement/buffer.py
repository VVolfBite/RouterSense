from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rs.core.contracts.measurement import MeasurementEvent, MeasurementSnapshot


@dataclass
class BoundedMeasurementBuffer:
    max_events: int
    _events: deque[MeasurementEvent] = field(init=False)
    _dropped: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if int(self.max_events) <= 0:
            raise ValueError("max_events must be > 0")
        self._events = deque(maxlen=int(self.max_events))

    def append(self, event: MeasurementEvent) -> None:
        if len(self._events) >= int(self.max_events):
            self._dropped += 1
        self._events.append(event)

    def snapshot(self, *, instrumentation_mode: str, summary: dict[str, object] | None = None) -> MeasurementSnapshot:
        return MeasurementSnapshot(
            event_count=len(self._events),
            dropped_event_count=int(self._dropped),
            instrumentation_mode=str(instrumentation_mode),
            events=tuple(self._events),
            summary=dict(summary or {}),
        )

    def reset(self) -> None:
        self._events.clear()
        self._dropped = 0
