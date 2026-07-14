from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rs.core.contracts.checks import CheckResult
from rs.core.contracts.debug import DebugEvent, DebugProbe
from rs.core.contracts.measurement import MeasurementEvent, MeasurementSink
from rs.core.contracts.result import ResultBundle
from rs.core.contracts.trace import ReferenceTraceBundle
from rs.runtime.debug.buffered_probe import BufferedDebugProbe, TensorCapture
from rs.runtime.debug.null_probe import NullDebugProbe
from rs.runtime.measurement.null_sink import NullMeasurementSink
from rs.runtime.measurement.perf_light import PerfLightMeasurementSink


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
class BufferedEvidenceSink:
    max_traces: int = 32
    max_results: int = 32
    traces: list[ReferenceTraceBundle] = field(default_factory=list)
    results: list[ResultBundle] = field(default_factory=list)
    dropped_trace_count: int = 0
    dropped_result_count: int = 0

    def record_trace(self, bundle: ReferenceTraceBundle) -> None:
        if len(self.traces) >= int(self.max_traces):
            self.traces.pop(0)
            self.dropped_trace_count += 1
        self.traces.append(bundle)

    def record_result(self, bundle: ResultBundle) -> None:
        if len(self.results) >= int(self.max_results):
            self.results.pop(0)
            self.dropped_result_count += 1
        self.results.append(bundle)

    def latest_result(self) -> ResultBundle | None:
        return self.results[-1] if self.results else None


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


def _normalize_instrumentation_mode(mode: str) -> str:
    normalized = str(mode or "off").strip().lower()
    legacy_map = {
        "minimal": "off",
        "execution": "contract",
        "perf": "perf_light",
        "timeline_light": "perf_light",
        "attribution_light": "perf_light",
    }
    return legacy_map.get(normalized, normalized or "off")


def build_runtime_instrumentation(
    *,
    instrumentation_mode: str,
    evidence_sink: EvidenceSink | None = None,
    measurement_capacity: int = 256,
    debug_capacity: int = 128,
) -> RuntimeInstrumentation:
    mode = _normalize_instrumentation_mode(instrumentation_mode)
    resolved_evidence_sink = evidence_sink or NullEvidenceSink()
    if mode == "off":
        return RuntimeInstrumentation(
            measurement_sink=NullMeasurementSink(),
            debug_probe=NullDebugProbe(),
            evidence_sink=resolved_evidence_sink,
        )
    if mode == "contract":
        return RuntimeInstrumentation(
            measurement_sink=PerfLightMeasurementSink(max_events=int(measurement_capacity)),
            debug_probe=NullDebugProbe(),
            evidence_sink=resolved_evidence_sink,
        )
    if mode == "perf_light":
        return RuntimeInstrumentation(
            measurement_sink=PerfLightMeasurementSink(max_events=int(measurement_capacity)),
            debug_probe=NullDebugProbe(),
            evidence_sink=resolved_evidence_sink,
        )
    if mode == "debug":
        return RuntimeInstrumentation(
            measurement_sink=PerfLightMeasurementSink(max_events=int(measurement_capacity)),
            debug_probe=BufferedDebugProbe(
                max_events=int(debug_capacity),
                capture=TensorCapture(enabled=True),
            ),
            evidence_sink=resolved_evidence_sink,
        )
    raise ValueError(f"unsupported instrumentation_mode: {instrumentation_mode!r}")
