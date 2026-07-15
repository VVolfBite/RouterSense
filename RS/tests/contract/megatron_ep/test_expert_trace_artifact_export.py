from __future__ import annotations

import json
from pathlib import Path

from rs.runtime.online.megatron_ep.phase_executor_artifacts import write_rank_artifacts


class _FakeRuntime:
    def export_prepared_plan_summary(self) -> dict:
        return {}

    def export_control_timeline(self) -> list[dict]:
        return []

    def export_control_commands(self) -> list[dict]:
        return []

    def export_assertions(self) -> dict:
        return {}

    def export_phase_contexts(self) -> list[dict]:
        return []

    def export_transport_bundles(self) -> list[dict]:
        return []

    def export_scheduled_phase_plans(self) -> list[dict]:
        return []

    def export_plan_arrival_records(self) -> list[dict]:
        return []

    def export_window_state_records(self) -> list[dict]:
        return []

    def export_prepared_plan_bindings(self) -> list[dict]:
        return []

    def export_release_events(self) -> list[dict]:
        return []

    def export_window_schedule_shadows(self) -> list[dict]:
        return []

    def export_prepared_phase_plan_shadows(self) -> list[dict]:
        return []

    def export_pending_window_driver_records(self) -> list[dict]:
        return []

    def export_planning_timing_records(self) -> list[dict]:
        return []

    def export_control_replay_traces(self) -> list[dict]:
        return []

    def export_prediction_audits(self) -> list[dict]:
        return []

    def export_expert_route_traces(self) -> list[dict]:
        return [{"layer_id": 1, "rank": 0}]

    def export_source_expert_counts(self) -> list[dict]:
        return [{"layer_id": 1, "rank": 0, "source_expert_counts": [[1, 0], [0, 0]]}]

    def export_expert_to_traffic_audits(self) -> list[dict]:
        return [{"layer_id": 1, "relative_l1_error": 0.0}]

    def export_expert_trace_warnings(self) -> list[dict]:
        return [{"layer_id": 1, "warning": "missing_routing_weights"}]

    def export_transport_execution_results(self) -> list[dict]:
        return []

    def export_captured_phase_tensors(self) -> list[dict]:
        return []

    def export_captured_phase_tensors_with_payload(self) -> list[dict]:
        return []


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_write_rank_artifacts_exports_expert_trace_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    summary = write_rank_artifacts(
        run_dir=run_dir,
        run_id="test-run",
        rank=0,
        logits=None,
        runtime=_FakeRuntime(),
        native_dispatch_summary={},
        rank_summary={},
        save_logits=False,
        capture_layer_selector="all",
        capture_phase_selector="both",
    )
    assert isinstance(summary, dict)
    route_path = run_dir / "rank0_expert_route_trace.jsonl"
    count_path = run_dir / "rank0_source_expert_counts.jsonl"
    audit_path = run_dir / "rank0_expert_to_traffic_audit.jsonl"
    warning_path = run_dir / "rank0_expert_trace_warnings.jsonl"
    for path in (route_path, count_path, audit_path, warning_path):
        assert path.exists()
    assert _read_jsonl(route_path) == [{"layer_id": 1, "rank": 0}]
    assert _read_jsonl(count_path)[0]["source_expert_counts"] == [[1, 0], [0, 0]]
    assert _read_jsonl(audit_path) == [{"layer_id": 1, "relative_l1_error": 0.0}]
    assert _read_jsonl(warning_path) == [{"layer_id": 1, "warning": "missing_routing_weights"}]
