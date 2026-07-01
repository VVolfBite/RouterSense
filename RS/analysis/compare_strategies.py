from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare greedy, fast, and oracle makespan fields.")
    parser.add_argument("--results-json", type=str, required=True)
    args = parser.parse_args(argv)
    rows = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    comparisons = []
    for row in rows:
        comparisons.append(
            {
                "sample_id": row["sample_id"],
                "layer_pair": row["layer_pair"],
                "greedy_makespan": row["greedy_makespan"],
                "fast_makespan": row.get("fast_makespan"),
                "oracle_perfect_makespan": row["oracle_perfect_makespan"],
                "oracle_predicted_makespan": row["oracle_predicted_makespan"],
            }
        )
    print(json.dumps(comparisons, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
