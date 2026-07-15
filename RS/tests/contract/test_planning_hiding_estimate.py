from __future__ import annotations

import json
from pathlib import Path

from rs.runtime.offline.planning_hiding import estimate_planning_hiding_window


def test_planning_hiding_estimate_cli(tmp_path: Path) -> None:
    summary_path = tmp_path / "planning_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary": [
                    {"mean_makespan": 2_000_000_000, "mean_prediction_relative_l1_error": 0.1},
                    {"mean_makespan": 1_500_000_000, "mean_prediction_relative_l1_error": 0.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = estimate_planning_hiding_window(
        planning_summary_path=summary_path,
        observed_forward_us=28_000_000,
        moe_layer_count=16,
    )
    assert "hide_feasible" in payload
    assert payload["estimated_hide_window_us"] > 0
