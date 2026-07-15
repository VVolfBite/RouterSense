from __future__ import annotations

import json
from pathlib import Path

from rs.runtime.offline.u_weight_tuning import run_u_weight_tuning


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


def test_u_weight_tuning_cli(tmp_path: Path) -> None:
    fixture_dir = _write_fixture_dir(tmp_path / "fixtures")
    payload = run_u_weight_tuning(
        fixture_dir=fixture_dir,
        policies=("U_barrier_criticality_global_matching",),
        grid_size=1,
    )
    assert "policies" in payload
    assert "U_barrier_criticality_global_matching" in payload["policies"]
    policy = payload["policies"]["U_barrier_criticality_global_matching"]
    assert float(policy["train_mean_makespan"]) > 0.0
    assert float(policy["eval_mean_makespan"]) > 0.0
    assert isinstance(policy["invalid_parameter_sets"], list)
