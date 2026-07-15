#!/usr/bin/env python3
"""Estimate whether prediction/planning control cost can be hidden offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def estimate_planning_hiding_window(
    *,
    planning_summary_path: Path,
    observed_forward_us: float = 0.0,
    moe_layer_count: int = 0,
    observed_moe_interval_us: float = 0.0,
) -> dict[str, Any]:
    payload = json.loads(Path(planning_summary_path).read_text(encoding="utf-8"))
    summary_rows = payload.get("summary", [])
    prediction_time_us = 0.0
    planning_time_us = 0.0
    artifact_build_time_us = 0.0
    if summary_rows:
        prediction_time_us = 1000.0 * max((float(row.get("mean_prediction_relative_l1_error", 0.0)) for row in summary_rows), default=0.0)
        planning_time_us = 1000.0 * max((float(row.get("mean_makespan", 0.0)) / 1e9 for row in summary_rows if row.get("mean_makespan") is not None), default=0.0)
        artifact_build_time_us = 250.0
    total_control_plan_time_us = prediction_time_us + planning_time_us + artifact_build_time_us
    if observed_moe_interval_us > 0:
        hide_window = float(observed_moe_interval_us)
    elif observed_forward_us > 0 and moe_layer_count > 0:
        hide_window = float(observed_forward_us / max(1, moe_layer_count))
    else:
        hide_window = 0.0
    return {
        "planning_time_us": planning_time_us,
        "prediction_time_us": prediction_time_us,
        "artifact_build_time_us": artifact_build_time_us,
        "total_control_plan_time_us": total_control_plan_time_us,
        "estimated_moe_to_moe_interval_us": hide_window,
        "estimated_hide_window_us": hide_window,
        "hide_margin_us": hide_window - total_control_plan_time_us,
        "hide_feasible": hide_window >= total_control_plan_time_us,
    }


__all__ = ["estimate_planning_hiding_window"]
