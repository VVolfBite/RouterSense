#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.evaluation import (
    build_owner_by_expert,
    build_same_prompt_batches,
    load_trace_jsonl,
    simulate_oracle_vs_greedy,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run POC-line1 oracle-vs-greedy offline simulation.")
    parser.add_argument("--trace-jsonl", type=str, required=True)
    parser.add_argument("--placement", type=str, default="round_robin", choices=["round_robin", "skewed"])
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--combine-scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "oracle_report"),
    )
    args = parser.parse_args(argv)

    records = load_trace_jsonl(args.trace_jsonl)
    owner_by_expert = build_owner_by_expert(records, placement=args.placement, num_gpus=args.num_gpus)
    batches = build_same_prompt_batches(records, owner_by_expert=owner_by_expert, num_gpus=args.num_gpus)
    report = simulate_oracle_vs_greedy(batches, combine_scale_factor=args.combine_scale_factor)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "placement.json", {"placement": args.placement, "owner_by_expert": owner_by_expert})
    write_json(out / "batches.json", batches)
    write_json(out / "results.json", report["results"])
    write_json(out / "summary.json", report["summary"])
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
