from __future__ import annotations

import json
from pathlib import Path

from experiments.offline.run_prediction_replay_suite import run_prediction_replay_suite


def _write_fixture_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    fixtures = [
        {
            "num_gpus": 2,
            "p0_dispatch_matrix": [[0, 8], [4, 0]],
            "p1_return_matrix": [[0, 4], [8, 0]],
            "p2_next_dispatch_forecast_matrix": [[0, 10], [5, 0]],
            "p2_next_dispatch_matrix": [[0, 10], [5, 0]],
            "metadata": {"layer_id": "0", "next_layer_id": "1"},
        },
        {
            "num_gpus": 2,
            "p0_dispatch_matrix": [[0, 10], [5, 0]],
            "p1_return_matrix": [[0, 5], [10, 0]],
            "p2_next_dispatch_forecast_matrix": [[0, 12], [6, 0]],
            "p2_next_dispatch_matrix": [[0, 12], [6, 0]],
            "metadata": {"layer_id": "1", "next_layer_id": "2"},
        },
    ]
    for idx, fixture in enumerate(fixtures):
        (path / f"replay_layer_{idx}.json").write_text(json.dumps(fixture), encoding="utf-8")
    return path


def test_prediction_replay_suite_cli(tmp_path: Path) -> None:
    fixture_dir = _write_fixture_dir(tmp_path / "fixtures")
    payload = run_prediction_replay_suite(
        fixture_dir=fixture_dir,
        policies=("RS_safe_barrier_criticality",),
        p2_sources=("zero_hint", "copy_current_dispatch", "fate_style_history"),
    )
    assert payload["summary"]
    assert any(row["policy_name"] == "RS_safe_barrier_criticality" for row in payload["summary"])
    assert any(row["p2_source"] == "zero_hint" for row in payload["summary"])
    assert payload["expert_trace_available"] is False
    assert payload["expert_trace_unavailable_reason"] == "expert_trace_unavailable_for_real_fixture"
