from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        paired_records.append(row)
    for row in payload.get("prediction", {}).get("records", []):
        paired_records.append(row)
    return {
        "sample_count": len(paired_records),
        "paired_records": paired_records,
        "groups": sorted(payload.keys()),
    }
