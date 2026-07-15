from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_real_trace_evidence_suite_cli(tmp_path: Path) -> None:
    replay_root = tmp_path / "bundle"
    fixture_dir = replay_root / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture0 = {
        "num_gpus": 4,
        "p0_dispatch_matrix": [[0, 32, 0, 16], [24, 0, 8, 0], [0, 12, 0, 20], [4, 0, 28, 0]],
        "p1_return_matrix": [[0, 24, 0, 4], [32, 0, 12, 0], [0, 8, 0, 28], [16, 0, 20, 0]],
        "p2_next_dispatch_forecast_matrix": [[0, 16, 0, 8], [12, 0, 6, 0], [0, 10, 0, 14], [2, 0, 18, 0]],
        "p2_next_dispatch_matrix": [[0, 16, 0, 8], [12, 0, 6, 0], [0, 10, 0, 14], [2, 0, 18, 0]],
        "metadata": {
            "layer_id": "0",
            "next_layer_id": "1",
            "p0_seen_ranks": [0, 1, 2, 3],
            "p1_seen_ranks": [0, 1, 2, 3],
            "p0_missing_ranks": [],
            "p1_missing_ranks": [],
            "p0_total_bytes": 144,
            "p1_total_bytes": 144,
            "p2_total_bytes": 86,
            "p2_source": "next_layer_p0_actual",
            "p0_nonzero_edge_count": 8,
            "p1_nonzero_edge_count": 8,
            "p2_nonzero_edge_count": 8,
        },
    }
    fixture1 = {
        "num_gpus": 4,
        "p0_dispatch_matrix": [[0, 20, 0, 12], [14, 0, 6, 0], [0, 18, 0, 10], [8, 0, 16, 0]],
        "p1_return_matrix": [[0, 18, 0, 8], [24, 0, 10, 0], [0, 6, 0, 20], [12, 0, 14, 0]],
        "p2_next_dispatch_forecast_matrix": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "p2_next_dispatch_matrix": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "metadata": {
            "layer_id": "1",
            "next_layer_id": "",
            "p0_seen_ranks": [0, 1, 2, 3],
            "p1_seen_ranks": [0, 1, 2, 3],
            "p0_missing_ranks": [],
            "p1_missing_ranks": [],
            "p0_total_bytes": 104,
            "p1_total_bytes": 112,
            "p2_total_bytes": 0,
            "p2_source": "zero_for_last_layer",
            "p0_nonzero_edge_count": 8,
            "p1_nonzero_edge_count": 8,
            "p2_nonzero_edge_count": 0,
        },
    }
    (fixture_dir / "replay_layer_0.json").write_text(json.dumps(fixture0), encoding="utf-8")
    (fixture_dir / "replay_layer_1.json").write_text(json.dumps(fixture1), encoding="utf-8")
    audit = {
        "source_kind": "control_replay_trace",
        "trace_file_count": 4,
        "policy_name": "birkhoff_phase_local",
        "run_id_digest": "abc123",
        "layer_count": 2,
        "fixture_count": 2,
        "num_gpus": 4,
        "expected_rank_count": 4,
        "layers": [fixture0["metadata"] | {"fixture_name": "replay_layer_0"}, fixture1["metadata"] | {"fixture_name": "replay_layer_1"}],
        "layer_count_with_complete_p0p1": 2,
        "layer_count_with_missing_rank": 0,
        "total_p0_bytes": 248,
        "total_p1_bytes": 256,
        "total_p2_bytes": 86,
        "avg_p0_bytes_per_layer": 124.0,
        "avg_p1_bytes_per_layer": 128.0,
        "avg_p2_bytes_per_layer": 43.0,
        "max_p0_bytes_layer": {"fixture_name": "replay_layer_0", "bytes": 144},
        "max_p1_bytes_layer": {"fixture_name": "replay_layer_0", "bytes": 144},
        "max_p2_bytes_layer": {"fixture_name": "replay_layer_0", "bytes": 86},
    }
    (replay_root / "replay_fixture_audit_summary.json").write_text(json.dumps(audit), encoding="utf-8")
    output_dir = tmp_path / "suite"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.run_real_trace_evidence_suite",
            "--fixture-dir",
            str(fixture_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    phase_sync = json.loads((output_dir / "phase_sync_compatible_summary.json").read_text(encoding="utf-8"))
    execution_window = json.loads((output_dir / "execution_window_joint_summary.json").read_text(encoding="utf-8"))
    prediction = json.loads((output_dir / "prediction_oracle_summary.json").read_text(encoding="utf-8"))
    paired = json.loads((output_dir / "paired_b_vs_u_summary.json").read_text(encoding="utf-8"))
    oracle_table = json.loads((output_dir / "oracle_table_summary.json").read_text(encoding="utf-8"))
    bridge = json.loads((output_dir / "bridge_candidates_summary.json").read_text(encoding="utf-8"))
    full = json.loads((output_dir / "real_trace_evidence_summary.json").read_text(encoding="utf-8"))
    assert phase_sync["baseline_policy"] == "birkhoff_phase_local"
    assert execution_window["baseline_policy"] == "B_birkhoff_wave"
    assert {row["p2_source"] for row in prediction["summary"]} == {
        "zero_hint",
        "copy_current_dispatch",
        "fate_style_history",
        "fate_style_linear",
        "perfect_trace_oracle",
        "actual_trace_oracle",
    }
    assert any(row["policy_name"] == "routersense_joint_priority_phase_sync" for row in bridge["summary"])
    assert any(row["policy_name"] == "routersense_joint_async_release_sim" for row in bridge["summary"])
    assert any(row["heuristic_family"] == "gated_maxweight_matching" for row in paired["summary"])
    assert oracle_table["summary"][0]["oracle_name"] == "O_local_phase_oracle"
    assert oracle_table["summary"][1]["oracle_name"] == "O_joint_cp_sat_oracle"
    assert full["pair_status"]["ready_pair_count"] >= 6
    assert full["best_pair"]["best_family"] in {
        "birkhoff_bvn",
        "gated_greedy",
        "gated_maxweight_matching",
        "barrier_criticality_matching",
        "barrier_price_adaptive_matching",
        "lagrangian_cross_phase",
    }
    markdown = (output_dir / "real_trace_evidence_summary.md").read_text(encoding="utf-8")
    assert "Paired B-vs-U result" in markdown
    assert "Joint scheduling space" in markdown
    assert "Cross-layer prediction value" in markdown
    assert "Oracle table" in markdown
    assert "RouterSense bridge candidates" in markdown
    assert "current RouterSense hint policy 不是 full joint execution-window scheduler" in markdown
