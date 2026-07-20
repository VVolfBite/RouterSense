from __future__ import annotations

import json
from pathlib import Path

from rs.core.contracts.result import RunIdentity
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle
from rs.reporting.strategy_overhead import run_overhead_audit


def _write_result_bundle(rep_dir: Path, *, policy_name: str, total_forward_us: int) -> None:
    bundle = build_result_bundle(
        ResultBundleDraft(
            run_identity=RunIdentity(
                run_id="run",
                pipeline="online",
                claim_scope="formal",
                trace_origin="runtime",
                future_information_mode="predicted",
            ),
            status="success",
            correctness_status="valid",
            performance_status="ineligible",
            commit_sha="abc123",
            git_clean=True,
            instrumentation_mode="contract",
            audit_evidence_level="summary_only",
            measurement_complete=True,
            summary={
                "all_work_completed": True,
                "fallback_count": 0,
                "timeout_count": 0,
                "check_failure_count": 0,
                "execution_outcome_count": 1,
                "total_forward_us": total_forward_us,
            },
            details={"policy_name": policy_name},
            extensions={},
        )
    )
    (rep_dir / "result_bundle.json").write_text(json.dumps(bundle.to_dict()), encoding="utf-8")


def test_overhead_report_does_not_label_hook_time_as_communication_makespan(tmp_path: Path) -> None:
    rep_dir = tmp_path / "run_a" / "birkhoff_bucket_phase_local" / "rep0"
    rep_dir.mkdir(parents=True)
    _write_result_bundle(rep_dir, policy_name="birkhoff_bucket_phase_local", total_forward_us=1000)
    planning_rows = [
        {"stage": "hook_before_token_dispatch_total", "duration_us": 100.0},
        {"stage": "hook_after_token_dispatch_total", "duration_us": 20.0},
        {"stage": "hook_before_token_combine_total", "duration_us": 80.0},
        {"stage": "hook_after_token_combine_total", "duration_us": 10.0},
        {"stage": "predict_next_dispatch", "duration_us": 30.0},
        {"stage": "run_phase_plan_agreement", "duration_us": 10.0},
    ]
    (rep_dir / "rank0_planning_timing.jsonl").write_text("\n".join(json.dumps(row) for row in planning_rows) + "\n", encoding="utf-8")
    (rep_dir / "rank0_phase_contexts.jsonl").write_text("", encoding="utf-8")
    (rep_dir / "rank0_transport_execution.jsonl").write_text(json.dumps({"phase": "dispatch"}) + "\n", encoding="utf-8")
    payload = run_overhead_audit(run_a_dir=tmp_path / "run_a", run_c_dir=None)
    row = payload["strategies"]["birkhoff_bucket_phase_local"]
    assert "communication_makespan_us" not in row
    assert row["transport_hook_path_total_us"] == 210.0
    assert row["actual_transport_makespan_us"] is None


def test_overhead_audit_skips_runs_without_canonical_result_bundle(tmp_path: Path) -> None:
    rep_dir = tmp_path / "run_a" / "birkhoff_bucket_phase_local" / "rep0"
    rep_dir.mkdir(parents=True)
    payload = run_overhead_audit(run_a_dir=tmp_path / "run_a", run_c_dir=None)
    assert payload["strategies"] == {}
