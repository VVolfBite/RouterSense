from __future__ import annotations

import json

from experiments.paper.aggregation import aggregate_results


def test_aggregation_keeps_negative_instances(tmp_path) -> None:
    (tmp_path / "scheduling_summary.json").write_text(
        json.dumps(
            {
                "records": [
                    {"instance_id": "a", "objective": -3.0, "comparable": True, "metadata": {"model_id": "m"}},
                    {"instance_id": "b", "objective": 2.0, "comparable": False, "metadata": {"model_id": "m"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "prediction_summary.json").write_text(
        json.dumps({"records": [{"instance_id": "c", "prediction_regret": None, "gain_over_zero": -0.25, "metadata": {"model_id": "m"}}]}),
        encoding="utf-8",
    )
    result = aggregate_results(input_dir=tmp_path)
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 1
    assert result["comparable_count"] == 1
    assert result["worst_case_objective"] == 2.0
    assert result["gain_over_zero"]["median"] == -0.25
