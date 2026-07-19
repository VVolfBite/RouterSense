from __future__ import annotations

import json

from rs.runtime.offline.prediction_policy_suite import run_prediction_suite


def _write_fixture(fixture_dir, layer_id: str, p0, p1, p2) -> None:
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
    (fixture_dir / f"replay_layer_{layer_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_prediction_suite_is_keyed_by_u_algorithm(tmp_path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    _write_fixture(
        fixture_dir,
        "0",
        [[0, 8, 0, 0], [0, 0, 0, 4], [6, 0, 0, 0], [0, 0, 2, 0]],
        [[0, 6, 0, 0], [8, 0, 0, 0], [0, 0, 0, 4], [0, 2, 0, 0]],
        [[0, 4, 0, 0], [0, 0, 0, 2], [3, 0, 0, 0], [0, 0, 1, 0]],
    )
    _write_fixture(
        fixture_dir,
        "1",
        [[0, 4, 0, 0], [0, 0, 0, 2], [3, 0, 0, 0], [0, 0, 1, 0]],
        [[0, 3, 0, 0], [4, 0, 0, 0], [0, 0, 0, 2], [0, 1, 0, 0]],
        [[0, 2, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]],
    )
    summary = run_prediction_suite(
        fixture_dir=fixture_dir,
        policies=("U_gated_maxweight_matching", "RS_safe_gated_maxweight"),
        p2_sources=("zero_hint", "copy_current_dispatch", "perfect_trace"),
        expert_compute_delay=0.0,
    )
    assert {row["U_algorithm"] for row in summary["summary"]} == {
        "U_gated_maxweight_matching",
        "RS_safe_gated_maxweight",
    }
    assert {row["p2_source"] for row in summary["summary"]} == {
        "zero_hint",
        "copy_current_dispatch",
        "perfect_trace_oracle",
    }
    assert any(row["heuristic_family"] == "gated_maxweight_matching" for row in summary["summary"])
    assert any(row["safe_policy"] is True for row in summary["summary"])
    assert all(row["predictor_name"] != "final_predictor" for row in summary["summary"])
