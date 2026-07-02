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
)
from routesense.scheduler import (  # noqa: E402
    fast_schedule_barrier_aware_birkhoff,
    fast_schedule_barrier_aware_birkhoff_wave,
    fast_schedule_birkhoff,
    fast_schedule_birkhoff_wave,
    fast_schedule_u_barrier_criticality_global_matching,
    fast_schedule_u_barrier_criticality_global_matching_atomic,
    fast_schedule_u_gated_maxweight_matching,
    fast_schedule_u_gated_maxweight_matching_atomic,
)


def _mean(summary: dict, key: str) -> float:
    value = summary.get(key, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0) or 0.0)
    return float(value or 0.0)


def _contribution_stats(summary: dict) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    decompositions = [
        (
            "gated_maxweight",
            "B_birkhoff",
            "B_birkhoff_wave",
            "U_gated_maxweight_matching_atomic",
            "U_gated_maxweight_matching",
        ),
        (
            "barrier_criticality",
            "B_barrier_aware_birkhoff",
            "B_barrier_aware_birkhoff_wave",
            "U_barrier_criticality_global_matching_atomic",
            "U_barrier_criticality_global_matching",
        ),
    ]
    for family, b_atomic_name, b_wave_name, u_atomic_name, u_wave_name in decompositions:
        b_atomic = _mean(summary, f"{b_atomic_name}_improvement_pct")
        b_wave = _mean(summary, f"{b_wave_name}_improvement_pct")
        u_atomic = _mean(summary, f"{u_atomic_name}_improvement_pct")
        u_wave = _mean(summary, f"{u_wave_name}_improvement_pct")
        total_gap = max(u_wave - b_atomic, 1e-9)
        rows.append(
            {
                "family": family,
                "b_atomic_pct": b_atomic,
                "b_wave_pct": b_wave,
                "u_atomic_pct": u_atomic,
                "u_wave_pct": u_wave,
                "b_wave_gain_pct": b_wave - b_atomic,
                "joint_gain_pct": u_atomic - b_wave,
                "u_wave_gain_pct": u_wave - u_atomic,
                "b_wave_share_pct": (b_wave - b_atomic) / total_gap * 100.0,
                "joint_share_pct": (u_atomic - b_wave) / total_gap * 100.0,
                "u_wave_share_pct": (u_wave - u_atomic) / total_gap * 100.0,
            }
        )
    return rows


def _markdown(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "| family | B_atomic% | B_wave% | U_atomic% | U_wave% | B_wave_gain% | joint_gain% | U_wave_gain% | B_wave_share% | joint_share% | U_wave_share% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {family} | {b_atomic_pct:.4f} | {b_wave_pct:.4f} | {u_atomic_pct:.4f} | {u_wave_pct:.4f} | {b_wave_gain_pct:.4f} | {joint_gain_pct:.4f} | {u_wave_gain_pct:.4f} | {b_wave_share_pct:.2f} | {joint_share_pct:.2f} | {u_wave_share_pct:.2f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablation: atomic joint scheduling vs fluid re-splitting.")
    parser.add_argument("--trace-jsonl", type=str, required=True)
    parser.add_argument("--hidden-states-path", type=str, required=True)
    parser.add_argument("--gate-weights-path", type=str, required=True)
    parser.add_argument("--placement", type=str, default="round_robin", choices=["round_robin", "skewed"])
    parser.add_argument("--num-gpus", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--phase2-source", choices=["predicted", "actual"], default="actual")
    parser.add_argument("--skip-oracle", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "poc_line1" / "ablation_fluid_vs_joint"))
    args = parser.parse_args(argv)

    records = load_trace_jsonl(args.trace_jsonl)
    hidden_states = load_hidden_state_bundle(args.hidden_states_path)
    gate_weights = load_gate_weight_bundle(args.gate_weights_path)
    owner_by_expert = build_owner_by_expert(records, placement=args.placement, num_gpus=args.num_gpus)

    selected_algorithms = [
        ("B_birkhoff", fast_schedule_birkhoff),
        ("B_birkhoff_wave", fast_schedule_birkhoff_wave),
        ("B_barrier_aware_birkhoff", fast_schedule_barrier_aware_birkhoff),
        ("B_barrier_aware_birkhoff_wave", fast_schedule_barrier_aware_birkhoff_wave),
        ("U_gated_maxweight_matching_atomic", fast_schedule_u_gated_maxweight_matching_atomic),
        ("U_gated_maxweight_matching", fast_schedule_u_gated_maxweight_matching),
        ("U_barrier_criticality_global_matching_atomic", fast_schedule_u_barrier_criticality_global_matching_atomic),
        ("U_barrier_criticality_global_matching", fast_schedule_u_barrier_criticality_global_matching),
    ]
    report = run_pairwise_analysis(
        records,
        hidden_states_by_sample=hidden_states,
        gate_weights_by_sample=gate_weights,
        owner_by_expert=owner_by_expert,
        num_gpus=args.num_gpus,
        topk=args.topk,
        sample_limit=args.sample_limit,
        phase2_source=args.phase2_source,
        fast_algorithms=selected_algorithms,
        skip_oracle=args.skip_oracle,
        include_fast_best_of=False,
    )

    rows = _contribution_stats(report["summary"])
    payload = {
        "config": {
            "num_gpus": args.num_gpus,
            "sample_limit": args.sample_limit,
            "phase2_source": args.phase2_source,
            "skip_oracle": args.skip_oracle,
        },
        "summary": report["summary"],
        "contribution_rows": rows,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "table.md").write_text(_markdown(rows) + "\n", encoding="utf-8")
    print(_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
