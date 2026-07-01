#!/usr/bin/env python3
from __future__ import annotations

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


CONFIGS = [
    {"name": "half_duplex", "model": "half_duplex", "expert_compute_delay": 0.0},
    {"name": "full_duplex", "model": "full_duplex", "expert_compute_delay": 0.0},
    {"name": "incast_only", "model": "incast_only", "expert_compute_delay": 0.0},
    {"name": "full_duplex_compute", "model": "full_duplex", "expert_compute_delay": 1.0},
]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Compare pairwise scheduling models.")
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--hidden-states-path", required=True)
    parser.add_argument("--gate-weights-path", required=True)
    parser.add_argument("--placement", default="round_robin", choices=["round_robin", "skewed"])
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=32,
        help="Prompt/sample cap for quick algorithm validation. Use small values like 32 or 64 by default; do not start with full batch500 unless explicitly needed.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    records = load_trace_jsonl(args.trace_jsonl)
    hidden_states = load_hidden_state_bundle(args.hidden_states_path)
    gate_weights = load_gate_weight_bundle(args.gate_weights_path)
    owner_by_expert = build_owner_by_expert(records, placement=args.placement, num_gpus=args.num_gpus)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    comparison_rows = []
    for config in CONFIGS:
        report = run_pairwise_analysis(
            records,
            hidden_states_by_sample=hidden_states,
            gate_weights_by_sample=gate_weights,
            owner_by_expert=owner_by_expert,
            num_gpus=args.num_gpus,
            topk=args.topk,
            sample_limit=args.sample_limit,
            model=config["model"],
            expert_compute_delay=float(config["expert_compute_delay"]),
        )
        reports[config["name"]] = report
        write_json(output_dir / f"{config['name']}_summary.json", report["summary"])
        comparison_rows.append(
            {
                "name": config["name"],
                "model": config["model"],
                "expert_compute_delay": config["expert_compute_delay"],
                "greedy_makespan_mean": sum(row["greedy_makespan"] for row in report["results"]) / max(len(report["results"]), 1),
                "oracle_perfect_makespan_mean": sum(row["oracle_perfect_makespan"] for row in report["results"]) / max(len(report["results"]), 1),
                "oracle_predicted_makespan_mean": sum(row["oracle_predicted_makespan"] for row in report["results"]) / max(len(report["results"]), 1),
                "fast_makespan_mean": sum(row["fast_makespan"] for row in report["results"]) / max(len(report["results"]), 1),
                "oracle_perfect_improvement_mean_pct": report["summary"]["perfect_improvement_pct"]["mean"],
                "oracle_predicted_improvement_mean_pct": report["summary"]["predicted_improvement_pct"]["mean"],
                "fast_improvement_mean_pct": report["summary"]["fast_improvement_pct"]["mean"],
                "fast_latency_mean_ms": report["summary"]["fast_latency_ms"]["mean"],
                "gate2_decision": report["summary"]["gate2_decision"]["decision"],
            }
        )

    write_json(output_dir / "comparison.json", comparison_rows)
    print(json.dumps(comparison_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
