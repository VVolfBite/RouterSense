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

from routesense.evaluation import (  # noqa: E402
    build_owner_by_expert,
    load_gate_weight_bundle,
    load_hidden_state_bundle,
    load_trace_jsonl,
    run_pairwise_analysis,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run POC-line1 pairwise predicted-oracle scheduling analysis.")
    parser.add_argument("--trace-jsonl", type=str, required=True)
    parser.add_argument("--hidden-states-path", type=str, required=True)
    parser.add_argument("--gate-weights-path", type=str, required=True)
    parser.add_argument("--placement", type=str, default="round_robin", choices=["round_robin", "skewed"])
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "pairwise_oracle_report"),
    )
    args = parser.parse_args(argv)

    records = load_trace_jsonl(args.trace_jsonl)
    hidden_states = load_hidden_state_bundle(args.hidden_states_path)
    gate_weights = load_gate_weight_bundle(args.gate_weights_path)
    owner_by_expert = build_owner_by_expert(records, placement=args.placement, num_gpus=args.num_gpus)
    report = run_pairwise_analysis(
        records,
        hidden_states_by_sample=hidden_states,
        gate_weights_by_sample=gate_weights,
        owner_by_expert=owner_by_expert,
        num_gpus=args.num_gpus,
        topk=args.topk,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "placement.json", {"placement": args.placement, "owner_by_expert": owner_by_expert})
    write_json(out / "results.json", report["results"])
    write_json(out / "summary.json", report["summary"])
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
