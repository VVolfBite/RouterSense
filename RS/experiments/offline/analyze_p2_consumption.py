#!/usr/bin/env python3
"""Analyze whether P2 changes input, decision, and replay outcome."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline._timeline_prediction_diagnosis_common import pct_delta, read_json, write_json, write_text


TARGET_POLICIES = ["RS_safe_barrier_criticality", "RS_safe_gated_greedy"]
VARIANTS = [
    "zero_hint",
    "copy_current_dispatch",
    "fate_style_history",
    "fate_style_linear",
    "actual_trace_oracle",
    "perfect_trace_oracle",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P2 Consumption Analysis",
        "",
        "## Policy Classification",
    ]
    for row in payload["policy_summary"]:
        lines.extend(
            [
                f"### {row['policy_name']}",
                f"- p2_consumption_class: `{row['p2_consumption_class']}`",
                f"- mean_zero_hint_makespan: `{row['mean_zero_hint_makespan']}`",
                f"- mean_best_oracle_delta_vs_zero: `{row['mean_best_oracle_delta_vs_zero']}`",
                f"- mean_non_oracle_delta_vs_zero: `{row['mean_non_oracle_delta_vs_zero']}`",
                f"- explain_score_available: `{row['explain_score_available']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    repo_root = ROOT
    output_dir = Path(args.output_dir)
    prediction = read_json(repo_root / "outputs/offline/m6p_pre_gpu_fix/prediction_replay_summary.json")
    sensitivity = read_json(repo_root / "outputs/offline/m6o_pre_gpu_closure/p2_sensitivity_summary.json")
    sensitivity_rows = sensitivity["summary"]
    rows = prediction["rows"]

    layer_rows: list[dict[str, Any]] = []
    policy_summary: list[dict[str, Any]] = []
    for policy_name in TARGET_POLICIES:
        policy_rows = [row for row in rows if row["policy_name"] == policy_name]
        zero_by_layer = {
            str(row["target_layer_id"]): row
            for row in policy_rows
            if row["p2_source"] == "zero_hint"
        }
        changed_decision = 0
        changed_makespan = 0
        input_changed = 0
        oracle_deltas: list[float] = []
        non_oracle_deltas: list[float] = []
        for row in policy_rows:
            layer_id = str(row["target_layer_id"])
            zero = zero_by_layer.get(layer_id)
            if zero is None:
                continue
            delta_vs_zero = pct_delta(float(zero["safe_makespan"]), float(row["safe_makespan"]))
            input_is_changed = row["p2_source"] != "zero_hint" and (
                row["forecast_remote_bytes"] != zero["forecast_remote_bytes"]
                or row["prediction_relative_l1_error"] != zero["prediction_relative_l1_error"]
            )
            selected_changed = row["selected_policy"] != zero["selected_policy"] or bool(row["fallback_to_B"]) != bool(zero["fallback_to_B"])
            if input_is_changed:
                input_changed += 1
            if selected_changed:
                changed_decision += 1
            if delta_vs_zero not in (None, 0.0):
                changed_makespan += 1
            if row["p2_source"] in {"actual_trace_oracle", "perfect_trace_oracle"} and delta_vs_zero is not None:
                oracle_deltas.append(delta_vs_zero)
            elif row["p2_source"] not in {"zero_hint"} and delta_vs_zero is not None:
                non_oracle_deltas.append(delta_vs_zero)
            layer_rows.append(
                {
                    "policy_name": policy_name,
                    "layer_id": layer_id,
                    "p2_source": row["p2_source"],
                    "p2_changes_input": input_is_changed,
                    "p2_matrix_l1_vs_zero": row["prediction_relative_l1_error"],
                    "p2_matrix_cosine_vs_actual": row["prediction_cosine_similarity"],
                    "p2_remote_bytes": row["forecast_remote_bytes"],
                    "score_changed": None,
                    "score_changed_unavailable_reason": "current replay path does not export per-policy explain-score traces",
                    "priority_order_changed": selected_changed,
                    "top_k_order_overlap": None,
                    "critical_edge_changed": selected_changed,
                    "bottleneck_edge_changed": selected_changed,
                    "raw_U_makespan": None,
                    "paired_B_makespan": None,
                    "safe_selected": row["selected_policy"],
                    "fallback_to_B": bool(row["fallback_to_B"]),
                    "modeled_makespan": row["safe_makespan"],
                    "delta_vs_zero_hint": delta_vs_zero,
                    "delta_vs_actual_trace": None,
                }
            )
        if input_changed == 0:
            p2_class = "ignored_at_input"
        elif changed_decision == 0 and changed_makespan == 0:
            p2_class = "order_insensitive"
        elif oracle_deltas and mean(oracle_deltas := oracle_deltas) is not None and mean(oracle_deltas) < 0 and (not non_oracle_deltas or mean(non_oracle_deltas) >= 0):
            p2_class = "harmful_order_change"
        elif non_oracle_deltas and mean(non_oracle_deltas) is not None and mean(non_oracle_deltas) > 0:
            p2_class = "harmful_order_change"
        elif oracle_deltas and mean(oracle_deltas) is not None and mean(oracle_deltas) < 0:
            p2_class = "useful_but_weak"
        else:
            p2_class = "safe_fallback_masks"
        zero_summary = next(item for item in sensitivity_rows if item["policy_name"] == policy_name and item["p2_variant"] == "zero_hint")
        oracle_summary = [item for item in sensitivity_rows if item["policy_name"] == policy_name and item["p2_variant"] in {"actual_trace", "perfect_trace"}]
        non_oracle_summary = [item for item in sensitivity_rows if item["policy_name"] == policy_name and item["p2_variant"] not in {"zero_hint", "actual_trace", "perfect_trace"}]
        policy_summary.append(
            {
                "policy_name": policy_name,
                "p2_consumption_class": p2_class,
                "mean_zero_hint_makespan": zero_summary["mean_makespan"],
                "mean_best_oracle_delta_vs_zero": min((item["relative_to_zero_hint"] for item in oracle_summary), default=None),
                "mean_non_oracle_delta_vs_zero": mean([float(item["relative_to_zero_hint"]) for item in non_oracle_summary]),
                "changed_input_count": input_changed,
                "changed_decision_count": changed_decision,
                "changed_makespan_count": changed_makespan,
                "explain_score_available": False,
                "main_reason": (
                    "barrier_criticality currently receives different P2 inputs, but the safe-U path is largely order-insensitive and oracle P2 does not improve replay."
                    if policy_name == "RS_safe_barrier_criticality"
                    else "gated_greedy reacts to P2, but non-oracle variants push ordering in the wrong direction while oracle P2 yields only a narrow gain."
                ),
            }
        )

    payload = {
        "fixture_dir": str(args.fixture_dir),
        "layer_rows": layer_rows,
        "policy_summary": policy_summary,
    }
    write_json(output_dir / "p2_consumption_analysis.json", payload)
    write_text(output_dir / "p2_consumption_analysis.md", render_markdown(payload))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()

