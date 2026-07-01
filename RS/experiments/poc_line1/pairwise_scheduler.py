#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Run POC-line1 pairwise predicted-oracle scheduling analysis.")
    parser.add_argument("--trace-jsonl", type=str, required=True)
    parser.add_argument("--hidden-states-path", type=str, required=True)
    parser.add_argument("--gate-weights-path", type=str, required=True)
    parser.add_argument("--placement", type=str, default="round_robin", choices=["round_robin", "skewed"])
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=32,
        help="Prompt/sample cap for quick scheduler validation. Recommended defaults are 32 or 64; avoid full batch500 until the algorithm has already passed small-sample checks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "pairwise_oracle_report"),
    )
    args = parser.parse_args(argv)

    print("[perf] loading trace...", flush=True)
    records = load_trace_jsonl(args.trace_jsonl)
    print(f"[perf] trace loaded: {time.time() - t0:.2f}s", flush=True)

    t_hidden = time.time()
    print("[perf] loading hidden states...", flush=True)
    hidden_states = load_hidden_state_bundle(args.hidden_states_path)
    print(f"[perf] hidden states loaded: {time.time() - t_hidden:.2f}s", flush=True)

    t_gates = time.time()
    print("[perf] loading gate weights...", flush=True)
    gate_weights = load_gate_weight_bundle(args.gate_weights_path)
    print(f"[perf] gate weights loaded: {time.time() - t_gates:.2f}s", flush=True)

    t_owner = time.time()
    owner_by_expert = build_owner_by_expert(records, placement=args.placement, num_gpus=args.num_gpus)
    print(f"[perf] owner_by_expert built: {time.time() - t_owner:.2f}s", flush=True)

    t_run = time.time()
    report = run_pairwise_analysis(
        records,
        hidden_states_by_sample=hidden_states,
        gate_weights_by_sample=gate_weights,
        owner_by_expert=owner_by_expert,
        num_gpus=args.num_gpus,
        topk=args.topk,
        sample_limit=args.sample_limit,
    )
    print(f"[perf] run_pairwise_analysis returned: {time.time() - t_run:.2f}s", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "placement.json", {"placement": args.placement, "owner_by_expert": owner_by_expert})
    write_json(out / "results.json", report["results"])
    write_json(out / "summary.json", report["summary"])
    write_json(
        out / "strategy_summary.json",
        {
            "baseline": "greedy_lpt",
            "fast_selection_rule": "best makespan among candidates with solve_time_ms <= 5.0, else global best",
            "fast_candidates": [
                "phase_aware_greedy",
                "birkhoff",
                "lagrangian",
                "iterated_greedy",
            ],
            "oracle_perfect": "cp_sat_true_next_dispatch",
            "oracle_predicted": "cp_sat_predicted_next_dispatch",
            "summary": report["summary"],
        },
    )
    print(f"[perf] total pairwise_scheduler main: {time.time() - t0:.2f}s", flush=True)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
