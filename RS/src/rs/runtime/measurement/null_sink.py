from __future__ import annotations

from rs.core.contracts.measurement import MeasurementEvent, MeasurementSnapshot


class NullMeasurementSink:
    def record(self, event: MeasurementEvent) -> None:
        event  # no-op

    def snapshot(self) -> MeasurementSnapshot:
        return MeasurementSnapshot(event_count=0, dropped_event_count=0, instrumentation_mode="off", events=(), summary={})

    def reset(self) -> None:
        return None
