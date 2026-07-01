from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump prediction-related pairwise fields for plotting.")
    parser.add_argument("--results-json", type=str, required=True)
    args = parser.parse_args(argv)
    rows = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    payload = [
        {
            "sample_id": row["sample_id"],
            "layer_pair": row["layer_pair"],
            "predicted_improvement_pct": row.get("predicted_improvement_pct", 0.0),
            "traffic_correlation": row.get("traffic_correlation", 0.0),
            "fast_improvement_pct": row.get("fast_improvement_pct", 0.0),
        }
        for row in rows
    ]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
