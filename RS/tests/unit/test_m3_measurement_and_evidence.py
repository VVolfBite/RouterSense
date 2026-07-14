from __future__ import annotations

from pathlib import Path

import torch

from rs.core.contracts.debug import DebugEvent
from rs.core.contracts.measurement import MeasurementEvent
from rs.core.contracts.result import ResultBundle, RunIdentity
from rs.core.contracts.trace import AuditEvidenceLevel, ReferenceTraceBundle
from rs.evidence.artifact_writer import FilesystemArtifactWriter
from rs.evidence.eligibility import evaluate_result_bundle_eligibility
from rs.evidence.serialization import EvidenceSerializer
from rs.runtime.debug.api import BufferedDebugProbe, NullDebugProbe, TensorCapture
from rs.runtime.measurement.api import NullMeasurementSink, PerfLightMeasurementSink


def test_null_measurement_sink_has_zero_events() -> None:
    sink = NullMeasurementSink()
    sink.record(MeasurementEvent(event_type="prediction", started_at_ns=1, ended_at_ns=2))
    snapshot = sink.snapshot()
    assert snapshot.event_count == 0
    assert snapshot.dropped_event_count == 0
    assert snapshot.instrumentation_mode == "off"


def test_perf_light_sink_is_bounded_and_filters_event_types() -> None:
    sink = PerfLightMeasurementSink(max_events=2)
    sink.record(MeasurementEvent(event_type="prediction", started_at_ns=1, ended_at_ns=2))
    sink.record(MeasurementEvent(event_type="planning", started_at_ns=2, ended_at_ns=3))
    sink.record(MeasurementEvent(event_type="unknown", started_at_ns=3, ended_at_ns=4))
    sink.record(MeasurementEvent(event_type="publish", started_at_ns=4, ended_at_ns=5))
    snapshot = sink.snapshot()
    assert snapshot.event_count == 2
    assert snapshot.dropped_event_count == 1
    assert tuple(event.event_type for event in snapshot.events) == ("planning", "publish")


def test_debug_probe_is_bounded() -> None:
    probe = BufferedDebugProbe(max_events=1, capture=TensorCapture(enabled=True, max_records=1))
    probe.record(DebugEvent(event_type="a", ts_ns=1, performance_eligible=False))
    probe.record(DebugEvent(event_type="b", ts_ns=2, performance_eligible=False))
    events = probe.flush()
    assert len(events) == 1
    assert events[0].event_type == "b"
    assert probe.dropped_event_count == 1
    assert NullDebugProbe().flush() == ()


def test_eligibility_fails_closed_for_empty_summary_and_debug() -> None:
    bundle = ResultBundle(
        run_identity=RunIdentity(
            run_id="run",
            pipeline="online",
            claim_scope="formal",
            trace_origin="observed",
            future_information_mode="predicted",
        ),
        status="success",
        eligibility=evaluate_result_bundle_eligibility(
            ResultBundle(
                run_identity=RunIdentity(
                    run_id="run",
                    pipeline="online",
                    claim_scope="formal",
                    trace_origin="observed",
                    future_information_mode="predicted",
                ),
                status="success",
                eligibility=None,  # type: ignore[arg-type]
                summary={},
                details={"instrumentation_mode": "debug"},
            )
        ),
        summary={},
        details={"instrumentation_mode": "debug"},
    )
    eligibility = evaluate_result_bundle_eligibility(bundle)
    assert eligibility.performance_eligible is False
    assert "empty_summary" in eligibility.reasons
    assert "debug_mode" in eligibility.reasons


def test_evidence_serializer_roundtrip_and_writer(tmp_path: Path) -> None:
    serializer = EvidenceSerializer()
    trace_bundle = ReferenceTraceBundle(
        run_identity={"run_id": "run"},
        topology={"world_size": 2},
        traffic_observations=({"layer_id": "0"},),
        evidence_level=AuditEvidenceLevel.SUMMARY_ONLY,
    )
    text = serializer.serialize_trace(trace_bundle)
    payload = serializer.deserialize_trace(text)
    assert payload["run_identity"]["run_id"] == "run"
    writer = FilesystemArtifactWriter(root_dir=tmp_path, serializer=serializer)
    output = writer.write_text(relative_path="trace.json", payload=text)
    assert Path(output).read_text(encoding="utf-8") == text


def test_measurement_and_debug_do_not_force_tensor_d2h(monkeypatch) -> None:
    calls = {"cpu": 0, "item": 0, "tolist": 0}

    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: calls.__setitem__("cpu", calls["cpu"] + 1) or self)
    monkeypatch.setattr(torch.Tensor, "item", lambda self, *a, **k: calls.__setitem__("item", calls["item"] + 1) or 0)
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self, *a, **k: calls.__setitem__("tolist", calls["tolist"] + 1) or [])

    sink = PerfLightMeasurementSink(max_events=4)
    sink.record(MeasurementEvent(event_type="prediction", started_at_ns=1, ended_at_ns=2))
    probe = BufferedDebugProbe(max_events=2)
    probe.record(DebugEvent(event_type="dbg", ts_ns=1, performance_eligible=False))
    sink.snapshot()
    probe.flush()
    assert calls == {"cpu": 0, "item": 0, "tolist": 0}
