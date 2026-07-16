from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[index])


def _virtual_ep_from_row(row: dict[str, Any]) -> str:
    value = row.get("virtual_ep_size")
    if value is not None:
        return str(value)
    instance_id = str(row.get("instance_id", ""))
    if ":vep" in instance_id:
        return instance_id.rsplit(":vep", 1)[-1]
    return "na"


def aggregate_results(*, input_dir: Path) -> dict[str, Any]:
    files = {
        "scheduling": input_dir / "scheduling_summary.json",
        "prediction": input_dir / "prediction_summary.json",
        "hiding": input_dir / "hiding_summary.json",
        "runtime": input_dir / "runtime_summary.json",
    }
    payload = {name: _load(path) for name, path in files.items() if path.exists()}
    paired_records: list[dict[str, Any]] = []
    for row in payload.get("scheduling", {}).get("records", []):
        paired_records.append({"record_type": "scheduling", **row})
    for row in payload.get("prediction", {}).get("records", []):
        paired_records.append({"record_type": "prediction", **row})
    valid = [row for row in paired_records if row.get("objective") is not None or row.get("prediction_regret") is not None]
    invalid = [row for row in paired_records if row not in valid]
    comparable = [row for row in paired_records if bool(row.get("comparable", False))]
    scheduling_objectives = [float(row["objective"]) for row in paired_records if row.get("record_type") == "scheduling" and row.get("objective") is not None]
    prediction_regrets = [float(row["prediction_regret"]) for row in paired_records if row.get("prediction_regret") is not None]
    gain_over_zero = [float(row["gain_over_zero"]) for row in paired_records if row.get("gain_over_zero") is not None]
    grouped: dict[str, dict[str, int]] = {}
    for row in paired_records:
        model = str(row.get("metadata", {}).get("model_id", "unknown"))
        layer = str(row.get("instance_id", "unknown")).split(":")[0]
        group_key = f"model={model}|layer={layer}|vep={_virtual_ep_from_row(row)}"
        grouped.setdefault(group_key, {"count": 0})
        grouped[group_key]["count"] += 1
    return {
        "sample_count": len(paired_records),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "comparable_count": len(comparable),
        "win_tie_loss": {"win": 0, "tie": 0, "loss": 0},
        "median_objective": None if not scheduling_objectives else float(median(scheduling_objectives)),
        "p90_objective": _percentile(scheduling_objectives, 0.90),
        "p95_objective": _percentile(scheduling_objectives, 0.95),
        "worst_case_objective": None if not scheduling_objectives else float(max(scheduling_objectives)),
        "oracle_gap": None,
        "prediction_regret": {
            "median": None if not prediction_regrets else float(median(prediction_regrets)),
            "p90": _percentile(prediction_regrets, 0.90),
        },
        "gain_over_zero": {
            "median": None if not gain_over_zero else float(median(gain_over_zero)),
            "p90": _percentile(gain_over_zero, 0.90),
        },
        "groups": grouped,
        "paired_records": paired_records,
    }
