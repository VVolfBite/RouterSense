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
    FAST_ALGORITHMS,
    build_owner_by_expert,
    load_gate_weight_bundle,
    load_hidden_state_bundle,
    load_trace_jsonl,
    run_pairwise_analysis,
)


def _stats_line(summary: dict, name: str) -> dict[str, float | None]:
    improve = summary.get(f"{name}_improvement_pct", {})
    latency = summary.get(f"{name}_latency_ms", {})
    return {
        "mean_improvement_pct": improve.get("mean"),
        "median_improvement_pct": improve.get("median"),
        "p75_improvement_pct": improve.get("p75"),
        "mean_latency_ms": latency.get("mean"),
    }


def _markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| algorithm | mean_improvement_pct | median_improvement_pct | p75_improvement_pct | mean_latency_ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {algorithm} | {mean_improvement_pct:.4f} | {median_improvement_pct:.4f} | {p75_improvement_pct:.4f} | {mean_latency_ms:.4f} |".format(
                algorithm=row["algorithm"],
                mean_improvement_pct=float(row["mean_improvement_pct"] or 0.0),
                median_improvement_pct=float(row["median_improvement_pct"] or 0.0),
                p75_improvement_pct=float(row["p75_improvement_pct"] or 0.0),
                mean_latency_ms=float(row["mean_latency_ms"] or 0.0),
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare selected pairwise schedulers on the same trace slice.")
    parser.add_argument("--trace-jsonl", type=str, required=True)
    parser.add_argument("--hidden-states-path", type=str, required=True)
    parser.add_argument("--gate-weights-path", type=str, required=True)
    parser.add_argument("--placement", type=str, default="round_robin", choices=["round_robin", "skewed"])
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument(
        "--algorithms",
        type=str,
        default="birkhoff,ibbr,ejection_chain_tabu,lns_cp_repair,decomposed,quantized_decomposed",
    )
    parser.add_argument("--skip-oracle", action="store_true", default=False)
    parser.add_argument("--skip-fast-best-of", action="store_true", default=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "candidate_compare"),
    )
    args = parser.parse_args(argv)

    records = load_trace_jsonl(args.trace_jsonl)
    hidden_states = load_hidden_state_bundle(args.hidden_states_path)
    gate_weights = load_gate_weight_bundle(args.gate_weights_path)
    owner_by_expert = build_owner_by_expert(records, placement=args.placement, num_gpus=args.num_gpus)

    by_name = {name: fn for name, fn in FAST_ALGORITHMS}
    selected_names = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown algorithms: {missing}")
    selected_algorithms = [(name, by_name[name]) for name in selected_names]

    report = run_pairwise_analysis(
        records,
        hidden_states_by_sample=hidden_states,
        gate_weights_by_sample=gate_weights,
        owner_by_expert=owner_by_expert,
        num_gpus=args.num_gpus,
        topk=args.topk,
        sample_limit=args.sample_limit,
        fast_algorithms=selected_algorithms,
        skip_oracle=args.skip_oracle,
        include_fast_best_of=not args.skip_fast_best_of,
    )

    rows = []
    for name in selected_names:
        row = {"algorithm": name}
        row.update(_stats_line(report["summary"], name))
        rows.append(row)
    if not args.skip_oracle:
        rows.append(
            {
                "algorithm": "oracle_perfect",
                "mean_improvement_pct": report["summary"]["perfect_improvement_pct"]["mean"],
                "median_improvement_pct": report["summary"]["perfect_improvement_pct"]["median"],
                "p75_improvement_pct": report["summary"]["perfect_improvement_pct"]["p75"],
                "mean_latency_ms": report["summary"]["oracle_perfect_latency_ms"]["mean"],
            }
        )
    table = _markdown_table(rows)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report["summary"], indent=2), encoding="utf-8")
    (out / "table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
