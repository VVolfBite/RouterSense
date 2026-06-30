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
    analyze_cross_layer_correlation,
    analyze_cross_layer_predictability,
    build_owner_by_expert,
    load_gate_weight_bundle,
    load_hidden_state_bundle,
    load_trace_jsonl,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run POC-line1 cross-layer correlation analysis.")
    parser.add_argument("--trace-jsonl", type=str, required=True)
    parser.add_argument("--hidden-states-path", type=str, default=None)
    parser.add_argument("--gate-weights-path", type=str, default=None)
    parser.add_argument("--placement", type=str, default="round_robin", choices=["round_robin", "skewed"])
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--skip-prediction-rows", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "cross_layer_report"),
    )
    args = parser.parse_args(argv)

    records = load_trace_jsonl(args.trace_jsonl)
    owner_by_expert = build_owner_by_expert(records, placement=args.placement, num_gpus=args.num_gpus)
    if args.hidden_states_path and args.gate_weights_path:
        hidden_states = load_hidden_state_bundle(args.hidden_states_path)
        gate_weights = load_gate_weight_bundle(args.gate_weights_path)
        topk = records[0].topk if records else 8
        report = analyze_cross_layer_predictability(
            records,
            hidden_states_by_sample=hidden_states,
            gate_weights_by_sample=gate_weights,
            topk=topk,
            owner_by_expert=owner_by_expert,
            num_gpus=args.num_gpus,
        )
    else:
        report = analyze_cross_layer_correlation(records, owner_by_expert=owner_by_expert, num_gpus=args.num_gpus)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in report.items():
        if args.skip_prediction_rows and name == "prediction_rows":
            continue
        write_json(out / f"{name}.json", payload)
    write_json(out / "placement.json", {"placement": args.placement, "owner_by_expert": owner_by_expert})
    print(json.dumps(report["gate1_decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
