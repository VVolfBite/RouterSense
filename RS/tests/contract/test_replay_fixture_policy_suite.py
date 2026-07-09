from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = "/root/autodl-tmp/RouterSense/RS"


def test_replay_fixture_policy_suite_cli(tmp_path: Path) -> None:
    replay_root = tmp_path / "bundle"
    fixture_dir = replay_root / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture = {
        "num_gpus": 4,
        "p0_dispatch_matrix": [[0, 32, 0, 16], [24, 0, 8, 0], [0, 12, 0, 20], [4, 0, 28, 0]],
        "p1_return_matrix": [[0, 24, 0, 4], [32, 0, 12, 0], [0, 8, 0, 28], [16, 0, 20, 0]],
        "p2_next_dispatch_forecast_matrix": [[0, 16, 0, 8], [12, 0, 6, 0], [0, 10, 0, 14], [2, 0, 18, 0]],
        "p2_next_dispatch_matrix": [[0, 16, 0, 8], [12, 0, 6, 0], [0, 10, 0, 14], [2, 0, 18, 0]],
        "metadata": {
            "layer_id": "0",
            "next_layer_id": "",
            "p0_seen_ranks": [0, 1, 2, 3],
            "p1_seen_ranks": [0, 1, 2, 3],
            "p0_missing_ranks": [],
            "p1_missing_ranks": [],
            "p0_total_bytes": 144,
            "p1_total_bytes": 144,
            "p2_total_bytes": 86,
            "p2_source": "zero_for_last_layer",
            "p0_nonzero_edge_count": 8,
            "p1_nonzero_edge_count": 8,
            "p2_nonzero_edge_count": 8,
        },
    }
    (fixture_dir / "replay_layer_0.json").write_text(json.dumps(fixture), encoding="utf-8")
    audit = {
        "source_kind": "control_replay_trace",
        "trace_file_count": 4,
        "policy_name": "disabled",
        "run_id_digest": "abc123",
        "layer_count": 1,
        "fixture_count": 1,
        "num_gpus": 4,
        "expected_rank_count": 4,
        "layers": [
            {
                "fixture_name": "replay_layer_0",
                "layer_id": "0",
                "next_layer_id": "",
                "p0_seen_ranks": [0, 1, 2, 3],
                "p1_seen_ranks": [0, 1, 2, 3],
                "p0_missing_ranks": [],
                "p1_missing_ranks": [],
                "p0_total_bytes": 144,
                "p1_total_bytes": 144,
                "p2_total_bytes": 86,
                "p2_source": "zero_for_last_layer",
                "p0_nonzero_edge_count": 8,
                "p1_nonzero_edge_count": 8,
                "p2_nonzero_edge_count": 8,
            }
        ],
        "layer_count_with_complete_p0p1": 1,
        "layer_count_with_missing_rank": 0,
        "total_p0_bytes": 144,
        "total_p1_bytes": 144,
        "total_p2_bytes": 86,
        "avg_p0_bytes_per_layer": 144.0,
        "avg_p1_bytes_per_layer": 144.0,
        "avg_p2_bytes_per_layer": 86.0,
        "max_p0_bytes_layer": {"fixture_name": "replay_layer_0", "bytes": 144},
        "max_p1_bytes_layer": {"fixture_name": "replay_layer_0", "bytes": 144},
        "max_p2_bytes_layer": {"fixture_name": "replay_layer_0", "bytes": 86},
    }
    (replay_root / "replay_fixture_audit_summary.json").write_text(json.dumps(audit), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_md_path = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.run_replay_fixture_policy_suite",
            "--fixture-dir",
            str(fixture_dir),
            "--output-summary",
            str(summary_path),
            "--output-summary-md",
            str(summary_md_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["table_a"]["mode"] == "runtime_lookahead"
    assert payload["table_b"]["mode"] == "execution_window"
    assert any(row["policy_name"] == "birkhoff_phase_local" for row in payload["table_a"]["summary"])
    assert any(row["policy_name"] == "B_birkhoff_wave" for row in payload["table_b"]["summary"])
    markdown = summary_md_path.read_text(encoding="utf-8")
    assert "Phase-sync-compatible result" in markdown
    assert "Execution-window joint result" in markdown
