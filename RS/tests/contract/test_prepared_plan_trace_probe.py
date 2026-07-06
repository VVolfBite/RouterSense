from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_prepared_plan_trace_probe_outputs_online_plan_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.online.probe_prepared_plan_trace",
            "--fixture",
            "tests/fixtures/scheduling/p2_lookahead_sensitive_4rank.json",
            "--output-dir",
            str(output_dir),
            "--run-id",
            "case",
            "--phase",
            "P0",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    run_dir = output_dir / "case"
    summary = json.loads((run_dir / "probe_summary.json").read_text(encoding="utf-8"))
    hint = json.loads((run_dir / "future_demand_hint.json").read_text(encoding="utf-8"))
    plan = json.loads((run_dir / "phase_execution_plan.json").read_text(encoding="utf-8"))
    audit = json.loads((run_dir / "synthetic_execution_audit.json").read_text(encoding="utf-8"))
    assert summary["compiled_from_prepared_plan"] is True
    assert summary["prepared_plan_order_preserved"] is True
    assert summary["hint_edges_consumed"] > 0
    assert hint["metadata"]["preferred_edges"]
    assert plan["metrics"]["compiled_from_prepared_plan"] is True
    assert audit["status"] == "passed"
