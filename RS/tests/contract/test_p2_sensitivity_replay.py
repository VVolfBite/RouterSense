from __future__ import annotations

import json

from experiments.offline.run_p2_sensitivity_replay import run_p2_sensitivity_replay


def _write_fixture(path, layer_id: str, p0, p1, p2) -> None:
    payload = {
        "num_gpus": 4,
        "p0_dispatch_matrix": p0,
        "p1_return_matrix": p1,
        "p2_next_dispatch_forecast_matrix": p2,
        "p2_next_dispatch_matrix": p2,
        "metadata": {
            "layer_id": layer_id,
            "next_layer_id": str(int(layer_id) + 1),
            "p0_seen_ranks": [0, 1, 2, 3],
            "p1_seen_ranks": [0, 1, 2, 3],
            "p0_missing_ranks": [],
            "p1_missing_ranks": [],
            "p0_total_bytes": 64,
            "p1_total_bytes": 64,
            "p2_total_bytes": 32,
            "p2_source": "next_layer_p0_actual",
            "p0_nonzero_edge_count": 4,
            "p1_nonzero_edge_count": 4,
            "p2_nonzero_edge_count": 4,
        },
    }
    (path / f"replay_layer_{layer_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_p2_sensitivity_replay_emits_expected_variants(tmp_path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    _write_fixture(
        fixture_dir,
        "0",
        [[0, 8, 0, 0], [0, 0, 0, 4], [6, 0, 0, 0], [0, 0, 2, 0]],
        [[0, 6, 0, 0], [8, 0, 0, 0], [0, 0, 0, 4], [0, 2, 0, 0]],
        [[0, 4, 0, 0], [0, 0, 0, 2], [3, 0, 0, 0], [0, 0, 1, 0]],
    )
    payload = run_p2_sensitivity_replay(fixture_dir=fixture_dir)
    variants = {row["p2_variant"] for row in payload["summary"]}
    assert "zero_hint" in variants
    assert "actual_trace" in variants
    assert "amplified_actual_4x" in variants
    assert "shuffled_actual" in variants
    assert "likely_reason_prediction_no_gain" in payload

