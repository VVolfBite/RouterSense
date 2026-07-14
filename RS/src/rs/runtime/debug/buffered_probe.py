from __future__ import annotations

from collections import deque

from rs.core.contracts.debug import DebugEvent


class TensorCapture:
    def __init__(self, *, enabled: bool, max_records: int = 32) -> None:
        self.enabled = bool(enabled)
        self.max_records = int(max_records)


class BufferedDebugProbe:
    def __init__(self, *, max_events: int = 128, capture: TensorCapture | None = None) -> None:
        if int(max_events) <= 0:
            raise ValueError("max_events must be > 0")
        self._events: deque[DebugEvent] = deque(maxlen=int(max_events))
        self._dropped = 0
        self.capture = capture or TensorCapture(enabled=False)

    @property
    def dropped_event_count(self) -> int:
        return int(self._dropped)

    def record(self, event: DebugEvent) -> None:
        if len(self._events) >= self._events.maxlen:
            self._dropped += 1
        self._events.append(event)

    def flush(self) -> tuple[DebugEvent, ...]:
        items = tuple(self._events)
        self._events.clear()
        return items
