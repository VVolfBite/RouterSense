from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from rs.core.contracts.debug import DebugEvent
from rs.core.contracts.measurement import MeasurementEvent
from rs.core.contracts.result import EligibilityResult, ResultBundle, RunIdentity
from rs.core.contracts.trace import AuditEvidenceLevel, ReferenceTraceBundle, TrafficObservationRecord
from rs.evidence.artifact_writer import FilesystemArtifactWriter
from rs.evidence.eligibility import evaluate_result_bundle_eligibility
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle
from rs.evidence.serialization import EvidenceSerializer
from rs.runtime.debug.api import BufferedDebugProbe, NullDebugProbe, TensorCapture
from rs.runtime.measurement.api import NullMeasurementSink, PerfLightMeasurementSink
from rs.runtime.observation.instrumentation import RuntimeInstrumentation
from rs.runtime.online.megatron_ep.observation.artifact_recorder import RuntimeArtifactRecorder
from rs.runtime.online.megatron_ep.observation.contracts import RuntimeObservationSnapshot


def _event(event_type: str, started_at_ns: int, ended_at_ns: int) -> MeasurementEvent:
    return MeasurementEvent(
        run_id="run",
        rank=0,
        forward_generation=0,
        microbatch_id="mb",
        event_type=event_type,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
    )


def _base_run_identity() -> RunIdentity:
    return RunIdentity(
        run_id="run",
        pipeline="online",
        claim_scope="formal",
        trace_origin="observed",
        future_information_mode="predicted",
    )


def _valid_result_bundle(*, instrumentation_mode: str = "off") -> ResultBundle:
    bundle = ResultBundle(
        run_identity=_base_run_identity(),
        status="success",
        correctness_status="valid",
        performance_status="eligible",
        pipeline="online",
        commit_sha="abc123",
        git_clean=True,
        instrumentation_mode=instrumentation_mode,
        audit_evidence_level="summary_only",
        measurement_complete=True,
        eligibility=EligibilityResult(
            correctness_eligible=True,
            performance_eligible=True,
            prediction_evaluation_eligible=True,
            offline_replay_eligible=True,
            preparation_claim_eligible=True,
            correctness_reasons=(),
            performance_reasons=(),
            prediction_reasons=(),
            offline_replay_reasons=(),
            preparation_claim_reasons=(),
        ),
        summary={
            "all_work_completed": True,
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 0,
            "execution_outcome_count": 1,
        },
        details={"backend": "cpu"},
        extensions={"lane": "formal"},
    )
    bundle.validate()
    return bundle


def test_null_measurement_sink_has_zero_events() -> None:
    sink = NullMeasurementSink()
    sink.record(_event("prediction", 1, 2))
    snapshot = sink.snapshot()
    assert snapshot.event_count == 0
    assert snapshot.dropped_event_count == 0
    assert snapshot.instrumentation_mode == "off"
    assert snapshot.completeness.complete is False


def test_perf_light_sink_is_bounded_and_filters_event_types() -> None:
    sink = PerfLightMeasurementSink(max_events=2)
    sink.record(_event("prediction", 1, 2))
    sink.record(_event("planning", 2, 3))
    sink.record(_event("unknown", 3, 4))
    sink.record(_event("publish", 4, 5))
    snapshot = sink.snapshot()
    assert snapshot.event_count == 2
    assert snapshot.unknown_event_count == 1
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


def test_result_bundle_rejects_reserved_extension_override() -> None:
    bundle = ResultBundle(
        run_identity=_base_run_identity(),
        status="success",
        correctness_status="valid",
        performance_status="eligible",
        pipeline="online",
        commit_sha="abc123",
        git_clean=True,
        instrumentation_mode="off",
        audit_evidence_level="summary_only",
        measurement_complete=True,
        eligibility=EligibilityResult(
            correctness_eligible=False,
            performance_eligible=False,
            prediction_evaluation_eligible=False,
            offline_replay_eligible=False,
            preparation_claim_eligible=False,
            correctness_reasons=("pending",),
            performance_reasons=("pending",),
            prediction_reasons=("pending",),
            offline_replay_reasons=("pending",),
            preparation_claim_reasons=("pending",),
        ),
        summary={
            "all_work_completed": True,
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 0,
            "execution_outcome_count": 1,
        },
        extensions={"pipeline": "bad"},
    )
    try:
        bundle.validate()
    except ValueError as exc:
        assert "reserved field" in str(exc)
    else:
        raise AssertionError("expected reserved extension validation failure")


def test_eligibility_fails_closed_for_empty_summary_and_debug() -> None:
    bundle = ResultBundle(
        run_identity=_base_run_identity(),
        status="success",
        correctness_status="valid",
        performance_status="eligible",
        pipeline="online",
        commit_sha="abc123",
        git_clean=True,
        instrumentation_mode="debug",
        audit_evidence_level="unavailable",
        measurement_complete=False,
        eligibility=EligibilityResult(
            correctness_eligible=False,
            performance_eligible=False,
            prediction_evaluation_eligible=False,
            offline_replay_eligible=False,
            preparation_claim_eligible=False,
            correctness_reasons=("pending",),
            performance_reasons=("pending",),
            prediction_reasons=("pending",),
            offline_replay_reasons=("pending",),
            preparation_claim_reasons=("pending",),
        ),
        summary={
            "all_work_completed": False,
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 0,
            "execution_outcome_count": 0,
        },
    )
    eligibility = evaluate_result_bundle_eligibility(bundle)
    assert eligibility.performance_eligible is False
    assert "debug_mode" in eligibility.correctness_reasons
    assert "measurement_incomplete" in eligibility.correctness_reasons


def test_result_bundle_roundtrip_returns_typed_bundle() -> None:
    serializer = EvidenceSerializer()
    bundle = _valid_result_bundle()
    payload = serializer.deserialize_result(serializer.serialize_result(bundle))
    assert isinstance(payload, ResultBundle)
    assert payload.run_identity.run_id == "run"
    assert payload.extensions["lane"] == "formal"


def test_eligibility_result_from_dict_rejects_string_booleans() -> None:
    try:
        EligibilityResult.from_dict(
            {
                "correctness_eligible": "false",
                "performance_eligible": False,
                "prediction_evaluation_eligible": False,
                "offline_replay_eligible": False,
                "preparation_claim_eligible": False,
                "correctness_reasons": (),
                "performance_reasons": (),
                "prediction_reasons": (),
                "offline_replay_reasons": (),
                "preparation_claim_reasons": (),
            }
        )
    except ValueError as exc:
        assert "correctness_eligible must be an explicit boolean" in str(exc)
    else:
        raise AssertionError("expected strict eligibility boolean validation failure")


def test_trace_bundle_roundtrip_returns_typed_bundle() -> None:
    serializer = EvidenceSerializer()
    trace_bundle = ReferenceTraceBundle(
        run_identity={"run_id": "run"},
        topology={"world_size": 2},
        traffic_observations=(
            TrafficObservationRecord(
                run_id="run",
                layer_id="1",
                phase="p0",
                layout_digest="layout",
                payload_roles=("hidden_states",),
            ),
        ),
        evidence_level=AuditEvidenceLevel.SUMMARY_ONLY,
    )
    payload = serializer.deserialize_trace(serializer.serialize_trace(trace_bundle))
    assert isinstance(payload, ReferenceTraceBundle)
    assert payload.traffic_observations[0].layer_id == "1"


def test_artifact_writer_roundtrip_and_path_guard(tmp_path: Path) -> None:
    serializer = EvidenceSerializer()
    writer = FilesystemArtifactWriter(root_dir=tmp_path, serializer=serializer)
    output = writer.write_text(relative_path="trace.json", payload="hello")
    assert Path(output).read_text(encoding="utf-8") == "hello"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["relative_path"] == "trace.json"
    try:
        writer.write_text(relative_path="../escape.txt", payload="bad")
    except ValueError as exc:
        assert "escapes root_dir" in str(exc)
    else:
        raise AssertionError("expected path escape validation failure")


def test_artifact_writer_jsonl_and_register_existing_roundtrip(tmp_path: Path) -> None:
    writer = FilesystemArtifactWriter(root_dir=tmp_path)
    writer.write_jsonl(
        relative_path="events.jsonl",
        payload=[{"a": 1}, {"b": 2}],
        schema="events.v1",
        producer="test",
        claim_role="diagnostic",
    )
    (tmp_path / "prebuilt.bin").write_bytes(b"abc")
    writer.register_existing(
        relative_path="prebuilt.bin",
        schema="binary.v1",
        producer="test",
        claim_role="raw",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {item["relative_path"] for item in manifest["artifacts"]}
    assert "events.jsonl" in artifact_paths
    assert "prebuilt.bin" in artifact_paths


def test_runtime_artifact_recorder_uses_canonical_manifest(tmp_path: Path) -> None:
    recorder = RuntimeArtifactRecorder(run_dir=tmp_path)
    recorder.write_run_manifest({"run_id": "run"})
    recorder.write_summary({"status": "success"})
    recorder.flush_snapshot(
        RuntimeObservationSnapshot(
            counters={"phase_context_count": 1},
            phase_contexts=({"layer_id": "1", "phase": "p0"},),
        )
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {item["relative_path"] for item in manifest["artifacts"]}
    assert "run_manifest.json" in artifact_paths
    assert "summary.json" in artifact_paths
    assert "observation_counters.json" in artifact_paths
    assert "phase_contexts.jsonl" in artifact_paths


def test_measurement_and_debug_do_not_force_tensor_d2h(monkeypatch) -> None:
    calls = {"cpu": 0, "item": 0, "tolist": 0}

    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: calls.__setitem__("cpu", calls["cpu"] + 1) or self)
    monkeypatch.setattr(torch.Tensor, "item", lambda self, *a, **k: calls.__setitem__("item", calls["item"] + 1) or 0)
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self, *a, **k: calls.__setitem__("tolist", calls["tolist"] + 1) or [])

    sink = PerfLightMeasurementSink(max_events=4)
    sink.record(_event("prediction", 1, 2))
    probe = BufferedDebugProbe(max_events=2)
    probe.record(DebugEvent(event_type="dbg", ts_ns=1, performance_eligible=False))
    sink.snapshot()
    probe.flush()
    assert calls == {"cpu": 0, "item": 0, "tolist": 0}


def test_runtime_instrumentation_off_and_perf_light_have_no_side_effects(monkeypatch, tmp_path: Path) -> None:
    calls = {
        "path_open": 0,
        "path_write_text": 0,
        "json_dump": 0,
        "json_dumps": 0,
        "cpu": 0,
        "item": 0,
        "tolist": 0,
    }

    original_open = Path.open
    original_write_text = Path.write_text
    original_json_dump = json.dump
    original_json_dumps = json.dumps

    monkeypatch.setattr(Path, "open", lambda self, *a, **k: calls.__setitem__("path_open", calls["path_open"] + 1) or original_open(self, *a, **k))
    monkeypatch.setattr(Path, "write_text", lambda self, *a, **k: calls.__setitem__("path_write_text", calls["path_write_text"] + 1) or original_write_text(self, *a, **k))
    monkeypatch.setattr(json, "dump", lambda *a, **k: calls.__setitem__("json_dump", calls["json_dump"] + 1) or original_json_dump(*a, **k))
    monkeypatch.setattr(json, "dumps", lambda *a, **k: calls.__setitem__("json_dumps", calls["json_dumps"] + 1) or original_json_dumps(*a, **k))
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k: calls.__setitem__("cpu", calls["cpu"] + 1) or self)
    monkeypatch.setattr(torch.Tensor, "item", lambda self, *a, **k: calls.__setitem__("item", calls["item"] + 1) or 0)
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self, *a, **k: calls.__setitem__("tolist", calls["tolist"] + 1) or [])

    off = RuntimeInstrumentation(measurement_sink=NullMeasurementSink(), debug_probe=NullDebugProbe())
    off.record_measurement(_event("prediction", 1, 2))
    off.record_debug(DebugEvent(event_type="dbg", ts_ns=1, performance_eligible=False))

    perf = RuntimeInstrumentation(
        measurement_sink=PerfLightMeasurementSink(max_events=4),
        debug_probe=NullDebugProbe(),
    )
    perf.record_measurement(_event("prediction", 2, 3))
    perf.measurement_sink.snapshot()

    assert calls["cpu"] == 0
    assert calls["item"] == 0
    assert calls["tolist"] == 0
    assert calls["path_open"] == 0
    assert calls["path_write_text"] == 0
    assert calls["json_dump"] == 0
    assert calls["json_dumps"] == 0


def test_debug_mode_is_not_performance_eligible() -> None:
    eligibility = evaluate_result_bundle_eligibility(_valid_result_bundle(instrumentation_mode="debug"))
    assert eligibility.performance_eligible is False
    assert "debug_mode" in eligibility.correctness_reasons


def test_prediction_and_offline_eligibility_require_domain_specific_fields() -> None:
    eligibility = evaluate_result_bundle_eligibility(_valid_result_bundle())
    assert eligibility.prediction_evaluation_eligible is False
    assert eligibility.offline_replay_eligible is False
    assert "prediction_evaluation_incomplete" in eligibility.prediction_reasons
    assert "offline_replay_incomplete" in eligibility.offline_replay_reasons


def test_prediction_and_offline_eligibility_pass_with_complete_summary() -> None:
    bundle = _valid_result_bundle(instrumentation_mode="perf_light")
    bundle.summary.update(
        {
            "run_kind": "OFFLINE_EVALUATION_FORMAL",
            "performance_measurement_complete": True,
            "measured_repeat_count": 1,
            "warmup_excluded": True,
            "prediction_evaluation_complete": True,
            "prediction_truth_digest": "pred-truth",
            "prediction_record_count": 2,
            "prediction_metric_count": 5,
            "prediction_audit_status": "valid",
            "truth_leakage_check": True,
            "offline_replay_complete": True,
            "evaluation_spec_digest": "spec",
            "task_set_digest": "taskset",
            "execution_truth_digest": "truth",
            "offline_record_count": 3,
            "offline_audit_status": "valid",
            "coverage_status": "complete",
        }
    )
    eligibility = evaluate_result_bundle_eligibility(bundle)
    assert eligibility.performance_eligible is True
    assert eligibility.prediction_evaluation_eligible is True
    assert eligibility.offline_replay_eligible is True


def test_result_bundle_performance_status_must_match_eligibility() -> None:
    bundle = replace(_valid_result_bundle(), measurement_complete=False)
    eligibility = evaluate_result_bundle_eligibility(bundle)
    assert eligibility.performance_eligible is False
    assert "performance_status_inconsistent" in eligibility.performance_reasons


def test_correctness_eligibility_fails_closed_for_incomplete_or_timed_out_runs() -> None:
    bundle = _valid_result_bundle()
    bundle = replace(
        bundle,
        summary={
            **bundle.summary,
            "all_work_completed": False,
            "timeout_count": 1,
            "check_failure_count": 1,
            "fallback_count": 1,
        },
        audit_evidence_level="unavailable",
    )
    eligibility = evaluate_result_bundle_eligibility(bundle)
    assert eligibility.correctness_eligible is False
    assert "all_work_incomplete" in eligibility.correctness_reasons
    assert "timeout_present" in eligibility.correctness_reasons
    assert "check_failures_present" in eligibility.correctness_reasons
    assert "audit_unavailable" in eligibility.correctness_reasons
    assert "correctness_status_inconsistent" in eligibility.reasons


def test_result_bundle_validate_rejects_summary_conflicts() -> None:
    bundle = replace(
        _valid_result_bundle(),
        summary={
            **_valid_result_bundle().summary,
            "status": "failure",
        },
    )
    try:
        bundle.validate()
    except ValueError as exc:
        assert "summary status conflicts" in str(exc)
    else:
        raise AssertionError("expected summary conflict validation failure")


def test_result_builder_rejects_missing_boolean_facts() -> None:
    try:
        build_result_bundle(
            ResultBundleDraft(
                run_identity=_base_run_identity(),
                status="success",
                correctness_status="valid",
                performance_status="ineligible",
                commit_sha="abc123",
                git_clean=None,  # type: ignore[arg-type]
                instrumentation_mode="contract",
                audit_evidence_level="summary_only",
                measurement_complete=None,  # type: ignore[arg-type]
                summary={
                    "all_work_completed": True,
                    "fallback_count": 0,
                    "timeout_count": 0,
                    "check_failure_count": 0,
                    "execution_outcome_count": 1,
                },
                details={},
                extensions={},
            )
        )
    except ValueError as exc:
        assert "explicit boolean" in str(exc)
    else:
        raise AssertionError("expected strict boolean fact validation failure")


def test_result_bundle_from_dict_requires_typed_eligibility_and_summary() -> None:
    bundle = _valid_result_bundle().to_dict()
    bundle.pop("eligibility")
    try:
        ResultBundle.from_dict(bundle)
    except ValueError as exc:
        assert "eligibility must be present and typed" in str(exc)
    else:
        raise AssertionError("expected missing eligibility failure")

    bundle = _valid_result_bundle().to_dict()
    bundle["summary"] = []
    try:
        ResultBundle.from_dict(bundle)
    except ValueError as exc:
        assert "summary must be present and typed" in str(exc)
    else:
        raise AssertionError("expected missing summary failure")
