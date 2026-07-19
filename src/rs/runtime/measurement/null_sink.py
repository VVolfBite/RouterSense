from __future__ import annotations

from rs.core.contracts.measurement import (
    MeasurementCapability,
    MeasurementCompleteness,
    MeasurementEvent,
    MeasurementSnapshot,
)


class NullMeasurementSink:
    def record(self, event: MeasurementEvent) -> None:
        event

    def snapshot(self) -> MeasurementSnapshot:
        return MeasurementSnapshot(
            event_count=0,
            dropped_event_count=0,
            instrumentation_mode="off",
            events=(),
            summary={},
            capability=MeasurementCapability(
                mode="off",
                emits_measurements=False,
                performance_claim_allowed=False,
            ),
            completeness=MeasurementCompleteness(
                complete=False,
                missing_required_event_types=(),
                unknown_event_count=0,
                malformed_event_count=0,
                dropped_event_count=0,
                overflowed=False,
            ),
            unknown_event_count=0,
            malformed_event_count=0,
        )

    def reset(self) -> None:
        return None
