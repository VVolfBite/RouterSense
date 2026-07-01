from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(fmean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize pairwise scheduling results.")
    parser.add_argument("--results-json", type=str, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    fields = ["fast_improvement_pct", "perfect_improvement_pct", "predicted_improvement_pct", "traffic_correlation"]
    summary = {field: _stats([float(row.get(field, 0.0)) for row in payload]) for field in fields}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
