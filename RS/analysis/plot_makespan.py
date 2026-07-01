from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump makespan tuples for downstream plotting.")
    parser.add_argument("--results-json", type=str, required=True)
    args = parser.parse_args(argv)
    rows = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    payload = [
        {
            "sample_id": row["sample_id"],
            "layer_pair": row["layer_pair"],
            "greedy_makespan": row["greedy_makespan"],
            "fast_makespan": row.get("fast_makespan", row["greedy_makespan"]),
            "oracle_perfect_makespan": row["oracle_perfect_makespan"],
            "oracle_predicted_makespan": row["oracle_predicted_makespan"],
            "greedy_latency_ms": row.get("greedy_latency_ms", 0.0),
            "fast_latency_ms": row.get("fast_latency_ms", 0.0),
            "oracle_perfect_latency_ms": row.get("oracle_perfect_latency_ms", 0.0),
            "oracle_predicted_latency_ms": row.get("oracle_predicted_latency_ms", 0.0),
        }
        for row in rows
    ]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
