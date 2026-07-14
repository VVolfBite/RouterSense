from __future__ import annotations

from rs.core.contracts.measurement import MeasurementEvent, MeasurementSnapshot
from rs.runtime.measurement.buffer import BoundedMeasurementBuffer


ALLOWED_EVENT_TYPES = {
    "forward",
    "selected_layer",
    "p0_hook",
    "p1_hook",
    "prediction",
    "planning",
    "publish",
    "materialization",
    "validation",
    "executor_submit",
    "executor_wait",
    "active_transport",
    "fallback",
    "check_counter",
}


class PerfLightMeasurementSink:
    def __init__(self, *, max_events: int = 256) -> None:
        self._buffer = BoundedMeasurementBuffer(max_events=max_events)

    def record(self, event: MeasurementEvent) -> None:
        if str(event.event_type) in ALLOWED_EVENT_TYPES:
            self._buffer.append(event)

    def snapshot(self) -> MeasurementSnapshot:
        summary = {
            "allowed_event_types": sorted(ALLOWED_EVENT_TYPES),
        }
        return self._buffer.snapshot(instrumentation_mode="perf_light", summary=summary)

    def reset(self) -> None:
        self._buffer.reset()
