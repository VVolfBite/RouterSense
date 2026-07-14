from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rs.core.contracts.checks import CheckResult
from rs.core.contracts.debug import DebugEvent, DebugProbe
from rs.core.contracts.measurement import MeasurementEvent, MeasurementSink
from rs.core.contracts.result import ResultBundle
from rs.core.contracts.trace import ReferenceTraceBundle


class EvidenceSink(Protocol):
    def record_trace(self, bundle: ReferenceTraceBundle) -> None:
        ...

    def record_result(self, bundle: ResultBundle) -> None:
        ...


class CheckRunner(Protocol):
    def run(self, *, stage: str, payload: dict[str, object]) -> tuple[CheckResult, ...]:
        ...


class NullEvidenceSink:
    def record_trace(self, bundle: ReferenceTraceBundle) -> None:
        bundle

    def record_result(self, bundle: ResultBundle) -> None:
        bundle


class NullCheckRunner:
    def run(self, *, stage: str, payload: dict[str, object]) -> tuple[CheckResult, ...]:
        stage
        payload
        return ()


@dataclass
class RuntimeInstrumentation:
    measurement_sink: MeasurementSink
    debug_probe: DebugProbe
    evidence_sink: EvidenceSink = field(default_factory=NullEvidenceSink)
    check_runner: CheckRunner = field(default_factory=NullCheckRunner)

    def record_measurement(self, event: MeasurementEvent) -> None:
        self.measurement_sink.record(event)

    def record_debug(self, event: DebugEvent) -> None:
        self.debug_probe.record(event)

    def record_trace(self, bundle: ReferenceTraceBundle) -> None:
        self.evidence_sink.record_trace(bundle)

    def record_result(self, bundle: ResultBundle) -> None:
        self.evidence_sink.record_result(bundle)
