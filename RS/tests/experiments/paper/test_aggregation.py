from __future__ import annotations

import json

from experiments.paper.aggregation import aggregate_results


def test_aggregation_keeps_negative_instances(tmp_path) -> None:
    (tmp_path / "scheduling_summary.json").write_text(json.dumps({"records": [{"instance_id": "a", "objective": 3.0}]}), encoding="utf-8")
    (tmp_path / "prediction_summary.json").write_text(json.dumps({"records": [{"instance_id": "b", "gain_over_zero": None}]}), encoding="utf-8")
    result = aggregate_results(input_dir=tmp_path)
    assert result["sample_count"] == 2
