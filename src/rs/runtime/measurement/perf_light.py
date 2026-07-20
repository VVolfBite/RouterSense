from __future__ import annotations

from rs.core.contracts.measurement import (
    MeasurementCapability,
    MeasurementCompleteness,
    MeasurementEvent,
    MeasurementRequirement,
    MeasurementSnapshot,
)
from rs.runtime.measurement.buffer import BoundedMeasurementBuffer


CONTRACT_EVENT_TYPES = {
    "planning",
    "publish",
    "materialization",
    "validation",
    "executor_submit",
    "executor_wait",
    "fallback",
    "check_counter",
}

PERF_LIGHT_EVENT_TYPES = CONTRACT_EVENT_TYPES | {
    "forward",
    "selected_layer",
    "p0_hook",
    "p1_hook",
    "prediction",
    "active_transport",
}


class _BaseMeasurementSink:
    def __init__(self, *, max_events: int, mode: str, allowed_event_types: set[str]) -> None:
        self._buffer = BoundedMeasurementBuffer(max_events=max_events)
        self._mode = str(mode)
        self._allowed_event_types = set(str(item) for item in allowed_event_types)

    def record(self, event: MeasurementEvent) -> None:
        try:
            event_type = str(event.event_type)
        except Exception:
            self._buffer.mark_malformed()
            return
        if event_type not in self._allowed_event_types:
            self._buffer.mark_unknown()
            return
        self._buffer.append(event)

    def _snapshot(self, *, requirement: MeasurementRequirement, performance_allowed: bool) -> MeasurementSnapshot:
        summary = {
            "allowed_event_types": sorted(self._allowed_event_types),
            "required_event_types": list(requirement.required_event_types),
        }
        provisional = self._buffer.snapshot(
            instrumentation_mode=self._mode,
            summary=summary,
            capability=MeasurementCapability(
                mode=self._mode,
                emits_measurements=True,
                performance_claim_allowed=bool(performance_allowed),
            ),
            completeness=MeasurementCompleteness(complete=False),
        )
        seen = {str(event.event_type) for event in provisional.events}
        missing_required = tuple(sorted(set(requirement.required_event_types) - seen))
        completeness = MeasurementCompleteness(
            complete=not missing_required
            and provisional.dropped_event_count == 0
            and provisional.unknown_event_count == 0
            and provisional.malformed_event_count == 0,
            missing_required_event_types=missing_required,
            unknown_event_count=provisional.unknown_event_count,
            malformed_event_count=provisional.malformed_event_count,
            dropped_event_count=provisional.dropped_event_count,
            overflowed=provisional.dropped_event_count > 0,
        )
        return self._buffer.snapshot(
            instrumentation_mode=self._mode,
            summary=summary,
            capability=MeasurementCapability(
                mode=self._mode,
                emits_measurements=True,
                performance_claim_allowed=bool(performance_allowed),
            ),
            completeness=completeness,
        )

    def reset(self) -> None:
        self._buffer.reset()


class ContractMeasurementSink(_BaseMeasurementSink):
    def __init__(self, *, max_events: int = 128) -> None:
        super().__init__(max_events=max_events, mode="contract", allowed_event_types=CONTRACT_EVENT_TYPES)

    def snapshot(self) -> MeasurementSnapshot:
        return self._snapshot(
            requirement=MeasurementRequirement(mode="contract", required_event_types=tuple(sorted(CONTRACT_EVENT_TYPES))),
            performance_allowed=False,
        )


class PerfLightMeasurementSink(_BaseMeasurementSink):
    def __init__(self, *, max_events: int = 256) -> None:
        super().__init__(max_events=max_events, mode="perf_light", allowed_event_types=PERF_LIGHT_EVENT_TYPES)

    def snapshot(self) -> MeasurementSnapshot:
        return self._snapshot(
            requirement=MeasurementRequirement(
                mode="perf_light",
                required_event_types=tuple(
                    sorted(
                        {
                            "planning",
                            "publish",
                            "materialization",
                            "validation",
                            "executor_submit",
                            "executor_wait",
                        }
                    )
                ),
                performance_claim_requested=True,
            ),
            performance_allowed=True,
        )
