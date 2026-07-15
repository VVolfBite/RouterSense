from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from rs.core.contracts.debug import DebugEvent


@dataclass
class TensorCapture:
    enabled: bool
    max_records: int = 32
    _records: deque[dict[str, Any]] = field(init=False)
    _lock: Lock = field(init=False, default_factory=Lock)

    def __post_init__(self) -> None:
        if int(self.max_records) <= 0:
            raise ValueError("max_records must be > 0")
        self._records = deque(maxlen=int(self.max_records))

    def record_metadata(self, *, label: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._records.append({"label": str(label), "payload": dict(payload)})

    def flush(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            items = tuple(self._records)
            self._records.clear()
        return items

    def reset(self) -> None:
        with self._lock:
            self._records.clear()


class BufferedDebugProbe:
    def __init__(self, *, max_events: int = 128, capture: TensorCapture | None = None) -> None:
        if int(max_events) <= 0:
            raise ValueError("max_events must be > 0")
        self._events: deque[DebugEvent] = deque(maxlen=int(max_events))
        self._dropped = 0
        self._lock = Lock()
        self.capture = capture or TensorCapture(enabled=False)

    @property
    def dropped_event_count(self) -> int:
        return int(self._dropped)

    def record(self, event: DebugEvent) -> None:
        with self._lock:
            if len(self._events) >= self._events.maxlen:
                self._dropped += 1
            self._events.append(event)

    def flush(self) -> tuple[DebugEvent, ...]:
        with self._lock:
            items = tuple(self._events)
            self._events.clear()
        return items
