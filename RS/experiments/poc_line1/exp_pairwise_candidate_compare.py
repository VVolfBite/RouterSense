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
from routesense.scheduler import fast_schedule_birkhoff_exhaustive  # noqa: E402


def _stats_line(summary: dict, name: str) -> dict[str, float | None]:
    improve = summary.get(f"{name}_improvement_pct", {})
    effective_improve = summary.get(f"{name}_effective_improvement_pct", {})
    latency = summary.get(f"{name}_latency_ms", {})
    return {
        "sched_improve_pct": improve.get("mean"),
        "effective_improve_pct": effective_improve.get("mean"),
        "prediction_aware": bool(summary.get(f"{name}_prediction_aware", False)),
        "mean_latency_ms": latency.get("mean"),
    }


def _e2e_stats_line(summary: dict, name: str) -> dict[str, float | None]:
    greedy = (summary.get("greedy_makespan_ms") or {}).get("mean")
    savings = (summary.get(f"{name}_comm_savings_ms") or {}).get("mean")
    return {
        "greedy_mksp_ms": greedy,
        "algo_mksp_ms": (None if greedy is None or savings is None else float(greedy) - float(savings)),
        "comm_savings_ms": savings,
        "sched_latency_ms": (summary.get(f"{name}_latency_ms") or {}).get("mean"),
        "exposed_ms": (summary.get(f"{name}_exposed_latency_ms") or {}).get("mean"),
        "net_benefit_ms": (summary.get(f"{name}_net_benefit_ms") or {}).get("mean"),
        "e2e_time_ms": (summary.get(f"{name}_effective_e2e_time_ms") or {}).get("mean"),
        "e2e_speedup_pct": (summary.get(f"{name}_e2e_speedup_pct") or {}).get("mean"),
    }


def _blind_rows(predicted_summary: dict, zeros_summary: dict, selected_names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in selected_names:
        predicted_imp = float((predicted_summary.get(f"{name}_improvement_pct") or {}).get("mean") or 0.0)
        zeros_imp = float((zeros_summary.get(f"{name}_improvement_pct") or {}).get("mean") or 0.0)
        delta = abs(predicted_imp - zeros_imp)
        declared = bool(predicted_summary.get(f"{name}_prediction_aware", False))
        actual = delta >= 1.0
        rows.append(
            {
                "algorithm": name,
                "declared_aware": declared,
                "predicted_imp_pct": predicted_imp,
                "zeros_imp_pct": zeros_imp,
                "delta_pct": delta,
                "actual_aware": actual,
                "match": declared == actual,
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| algorithm | sched_improve% | effective_improve% | prediction_aware | latency_ms |",
        "|---|---:|---:|:---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {algorithm} | {sched_improve_pct:.4f} | {effective_improve_pct:.4f} | {prediction_aware} | {mean_latency_ms:.4f} |".format(
                algorithm=row["algorithm"],
                sched_improve_pct=float(row["sched_improve_pct"] or 0.0),
                effective_improve_pct=float(row["effective_improve_pct"] or 0.0),
                prediction_aware="yes" if bool(row["prediction_aware"]) else "no",
                mean_latency_ms=float(row["mean_latency_ms"] or 0.0),
            )
        )
    return "\n".join(lines)


def _blind_markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| algorithm | declared_aware | predicted_imp% | zeros_imp% | delta% | actual_aware | match |",
        "|---|:---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {algorithm} | {declared} | {predicted:.4f} | {zeros:.4f} | {delta:.4f} | {actual} | {match} |".format(
                algorithm=row["algorithm"],
                declared="yes" if bool(row["declared_aware"]) else "no",
                predicted=float(row["predicted_imp_pct"]),
                zeros=float(row["zeros_imp_pct"]),
                delta=float(row["delta_pct"]),
                actual="yes" if bool(row["actual_aware"]) else "no",
                match="yes" if bool(row["match"]) else "no",
            )
        )
    return "\n".join(lines)


def _e2e_markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| algorithm | greedy_mksp | algo_mksp | comm_savings_ms | sched_latency_ms | exposed_ms | net_benefit_ms | e2e_time_ms | e2e_speedup% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {algorithm} | {greedy:.4f} | {algo:.4f} | {savings:.4f} | {latency:.4f} | {exposed:.4f} | {benefit:.4f} | {e2e:.4f} | {speedup:.4f} |".format(
                algorithm=row["algorithm"],
                greedy=float(row["greedy_mksp_ms"] or 0.0),
                algo=float(row["algo_mksp_ms"] or 0.0),
                savings=float(row["comm_savings_ms"] or 0.0),
                latency=float(row["sched_latency_ms"] or 0.0),
                exposed=float(row["exposed_ms"] or 0.0),
                benefit=float(row["net_benefit_ms"] or 0.0),
                e2e=float(row["e2e_time_ms"] or 0.0),
                speedup=float(row["e2e_speedup_pct"] or 0.0),
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
        default="B_birkhoff,B_barrier_aware_birkhoff,O_cp_lpt,O_lagrangian,O_ibbr,O_gated_greedy_maximal,O_gated_maxweight_matching,O_barrier_criticality_global_matching,O_barrier_price_adaptive_matching",
    )
    parser.add_argument(
        "--phase2-source",
        choices=["predicted", "actual"],
        default="predicted",
        help="Which phase-2 matrix the compared schedulers actually optimize against.",
    )
    parser.add_argument("--skip-oracle", action="store_true", default=False)
    parser.add_argument("--skip-fast-best-of", action="store_true", default=True)
    parser.add_argument("--hidden-window-ms", type=float, default=10.0)
    parser.add_argument("--token-to-ms-factor", type=float, default=0.5)
    parser.add_argument(
        "--next-mode",
        choices=["predicted", "zeros", "both"],
        default="predicted",
        help="next_dispatch_matrix mode: predicted, zeros, or both",
    )
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
    by_name["birkhoff_exhaustive"] = fast_schedule_birkhoff_exhaustive
    selected_names = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown algorithms: {missing}")
    selected_algorithms = [(name, by_name[name]) for name in selected_names]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def run_and_collect(mode: str) -> tuple[dict, str]:
        report = run_pairwise_analysis(
            records,
            hidden_states_by_sample=hidden_states,
            gate_weights_by_sample=gate_weights,
            owner_by_expert=owner_by_expert,
            num_gpus=args.num_gpus,
            topk=args.topk,
            sample_limit=args.sample_limit,
            hidden_window_ms=args.hidden_window_ms,
            token_to_ms_factor=args.token_to_ms_factor,
            next_mode=mode,
            phase2_source=args.phase2_source,
            fast_algorithms=selected_algorithms,
            skip_oracle=args.skip_oracle,
            include_fast_best_of=not args.skip_fast_best_of,
        )
        rows = []
        e2e_rows = []
        for name in selected_names:
            row = {"algorithm": name}
            row.update(_stats_line(report["summary"], name))
            rows.append(row)
            e2e_row = {"algorithm": name}
            e2e_row.update(_e2e_stats_line(report["summary"], name))
            e2e_rows.append(e2e_row)
        if not args.skip_oracle:
            rows.append(
                {
                    "algorithm": "oracle_perfect",
                    "sched_improve_pct": report["summary"]["perfect_improvement_pct"]["mean"],
                    "effective_improve_pct": report["summary"]["perfect_improvement_pct"]["mean"],
                    "prediction_aware": False,
                    "mean_latency_ms": report["summary"]["oracle_perfect_latency_ms"]["mean"],
                }
            )
            oracle_e2e = {
                "algorithm": "oracle_perfect",
                "greedy_mksp_ms": report["summary"]["greedy_makespan_ms"]["mean"],
                "algo_mksp_ms": report["summary"]["oracle_perfect_effective_e2e_time_ms"]["mean"] - report["summary"]["oracle_perfect_exposed_latency_ms"]["mean"],
                "comm_savings_ms": report["summary"]["oracle_perfect_comm_savings_ms"]["mean"],
                "sched_latency_ms": report["summary"]["oracle_perfect_latency_ms"]["mean"],
                "exposed_ms": report["summary"]["oracle_perfect_exposed_latency_ms"]["mean"],
                "net_benefit_ms": report["summary"]["oracle_perfect_net_benefit_ms"]["mean"],
                "e2e_time_ms": report["summary"]["oracle_perfect_effective_e2e_time_ms"]["mean"],
                "e2e_speedup_pct": report["summary"]["oracle_perfect_e2e_speedup_pct"]["mean"],
            }
            e2e_rows.append(oracle_e2e)
        e2e_rows.insert(
            0,
            {
                "algorithm": "greedy (no sched)",
                "greedy_mksp_ms": report["summary"]["greedy_makespan_ms"]["mean"],
                "algo_mksp_ms": report["summary"]["greedy_makespan_ms"]["mean"],
                "comm_savings_ms": 0.0,
                "sched_latency_ms": 0.0,
                "exposed_ms": 0.0,
                "net_benefit_ms": 0.0,
                "e2e_time_ms": report["summary"]["greedy_makespan_ms"]["mean"],
                "e2e_speedup_pct": 0.0,
            },
        )
        return report, _markdown_table(rows), _e2e_markdown_table(e2e_rows)

    if args.next_mode != "both":
        report, table, e2e_table = run_and_collect(args.next_mode)
        (out / "summary.json").write_text(json.dumps(report["summary"], indent=2), encoding="utf-8")
        (out / "table.md").write_text(table + "\n", encoding="utf-8")
        (out / "table_e2e.md").write_text(e2e_table + "\n", encoding="utf-8")
        print(table)
        print()
        print(e2e_table)
        return 0

    predicted_report, predicted_table, predicted_e2e_table = run_and_collect("predicted")
    zeros_report, zeros_table, zeros_e2e_table = run_and_collect("zeros")
    blind_rows = _blind_rows(predicted_report["summary"], zeros_report["summary"], selected_names)
    prediction_blind_test = {
        row["algorithm"]: {
            "predicted_improvement_pct": row["predicted_imp_pct"],
            "zeros_improvement_pct": row["zeros_imp_pct"],
            "delta_pct": row["delta_pct"],
            "prediction_aware": row["actual_aware"],
            "prediction_aware_declared": row["declared_aware"],
            "match": row["match"],
        }
        for row in blind_rows
    }
    combined_summary = {
        "predicted": predicted_report["summary"],
        "zeros": zeros_report["summary"],
        "prediction_blind_test": prediction_blind_test,
        "prediction_aware_algorithms": [row["algorithm"] for row in blind_rows if bool(row["actual_aware"])],
        "prediction_blind_algorithms": [row["algorithm"] for row in blind_rows if not bool(row["actual_aware"])],
        "mismatched": [row["algorithm"] for row in blind_rows if not bool(row["match"])],
    }
    blind_table = _blind_markdown_table(blind_rows)
    (out / "summary_predicted.json").write_text(json.dumps(predicted_report["summary"], indent=2), encoding="utf-8")
    (out / "summary_zeros.json").write_text(json.dumps(zeros_report["summary"], indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    (out / "table_predicted.md").write_text(predicted_table + "\n", encoding="utf-8")
    (out / "table_zeros.md").write_text(zeros_table + "\n", encoding="utf-8")
    (out / "table_predicted_e2e.md").write_text(predicted_e2e_table + "\n", encoding="utf-8")
    (out / "table_zeros_e2e.md").write_text(zeros_e2e_table + "\n", encoding="utf-8")
    (out / "table.md").write_text(blind_table + "\n", encoding="utf-8")
    print(blind_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
