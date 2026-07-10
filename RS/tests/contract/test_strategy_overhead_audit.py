from __future__ import annotations

import json
from pathlib import Path

from experiments.online.analyze_4gpu_strategy_overhead import run_overhead_audit


def test_overhead_report_does_not_label_hook_time_as_communication_makespan(tmp_path: Path) -> None:
    rep_dir = tmp_path / "run_a" / "birkhoff_phase_local" / "rep0"
    rep_dir.mkdir(parents=True)
    (rep_dir / "summary.json").write_text(json.dumps({"status": "ready", "policy_name": "birkhoff_phase_local", "total_forward_us": 1000}), encoding="utf-8")
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
    row = payload["strategies"]["birkhoff_phase_local"]
    assert "communication_makespan_us" not in row
    assert row["transport_hook_path_total_us"] == 210.0
    assert row["actual_transport_makespan_us"] is None
