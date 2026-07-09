from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_transport_stress_replay_cli(tmp_path: Path) -> None:
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
            "experiments.offline.run_transport_stress_replay",
            "--fixture-dir",
            str(fixture_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    summary = json.loads((output_dir / "transport_stress_replay_summary.json").read_text(encoding="utf-8"))
    assert "phase_sync_transport" in summary
    assert "joint_transport" in summary
    assert summary["phase_sync_transport"]["baseline_policy"] == "birkhoff_phase_local"
    assert summary["joint_transport"]["baseline_policy"] == "B_birkhoff_wave"
    markdown = (output_dir / "transport_stress_replay_summary.md").read_text(encoding="utf-8")
    assert "communication-only" in markdown
    assert "Joint upper-bound transport replay" in markdown
