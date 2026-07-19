from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from rs.core.contracts.measurement import (
    MeasurementCapability,
    MeasurementCompleteness,
    MeasurementEvent,
    MeasurementSnapshot,
)


@dataclass
class BoundedMeasurementBuffer:
    max_events: int
    _events: deque[MeasurementEvent] = field(init=False)
    _dropped: int = field(init=False, default=0)
    _unknown: int = field(init=False, default=0)
    _malformed: int = field(init=False, default=0)
    _lock: Lock = field(init=False, default_factory=Lock)

    def __post_init__(self) -> None:
        if int(self.max_events) <= 0:
            raise ValueError("max_events must be > 0")
        self._events = deque(maxlen=int(self.max_events))

    def append(self, event: MeasurementEvent) -> None:
        with self._lock:
            if len(self._events) >= int(self.max_events):
                self._dropped += 1
            self._events.append(event)

    def mark_unknown(self) -> None:
        with self._lock:
            self._unknown += 1

    def mark_malformed(self) -> None:
        with self._lock:
            self._malformed += 1

    def snapshot(
        self,
        *,
        instrumentation_mode: str,
        summary: dict[str, object] | None = None,
        capability: MeasurementCapability,
        completeness: MeasurementCompleteness,
    ) -> MeasurementSnapshot:
        with self._lock:
            events = tuple(self._events)
            dropped = int(self._dropped)
            unknown = int(self._unknown)
            malformed = int(self._malformed)
        return MeasurementSnapshot(
            event_count=len(events),
            dropped_event_count=dropped,
            instrumentation_mode=str(instrumentation_mode),
            events=events,
            summary=dict(summary or {}),
            capability=capability,
            completeness=completeness,
            unknown_event_count=unknown,
            malformed_event_count=malformed,
        )

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._dropped = 0
            self._unknown = 0
            self._malformed = 0
