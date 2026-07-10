#!/usr/bin/env python3
"""Layer-by-layer decision diff for safe-U families."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline._timeline_prediction_diagnosis_common import (
    flatten_topology_signature,
    pct_delta,
    read_json,
    write_json,
    write_text,
)


SAFE_POLICY_FAMILIES = {
    "RS_safe_barrier_criticality": {
        "family": "barrier_criticality_matching",
        "paired_B": "B_barrier_criticality_matching",
        "raw_U": "U_barrier_criticality_global_matching",
    },
    "RS_safe_gated_greedy": {
        "family": "gated_greedy",
        "paired_B": "B_gated_greedy_maximal",
        "raw_U": "U_gated_greedy_maximal",
    },
    "RS_safe_gated_maxweight": {
        "family": "gated_maxweight_matching",
        "paired_B": "B_gated_maxweight_matching",
        "raw_U": "U_gated_maxweight_matching",
    },
    "RS_safe_barrier_price": {
        "family": "barrier_price_adaptive_matching",
        "paired_B": "B_barrier_price_adaptive_matching",
        "raw_U": "U_barrier_price_adaptive_matching",
    },
    "RS_safe_ibbr": {
        "family": "birkhoff_bvn",
        "paired_B": "B_birkhoff",
        "raw_U": "U_ibbr",
    },
    "RS_safe_lagrangian": {
        "family": "lagrangian_cross_phase",
        "paired_B": "B_lagrangian_phase_local",
        "raw_U": "U_lagrangian",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _p2_matrix_for_variant(fixture: dict[str, Any], p2_source: str) -> list[list[int]] | None:
    if p2_source == "zero_hint":
        size = int(fixture["num_gpus"])
        return [[0 for _ in range(size)] for _ in range(size)]
    if p2_source == "copy_current_dispatch":
        return fixture["p0_dispatch_matrix"]
    if p2_source in {"perfect_trace_oracle", "actual_trace_oracle"}:
        return fixture["p2_next_dispatch_matrix"]
    if p2_source in {"actual_trace", "perfect_trace"}:
        return fixture["p2_next_dispatch_matrix"]
    return None


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Safe-U Decision Diff",
        "",
        "## Summary",
    ]
    for policy_name, info in payload["summary"].items():
        lines.extend(
            [
                f"### {policy_name}",
                f"- benefit_layers: `{info['benefit_layers']}`",
                f"- harm_layers: `{info['harm_layers']}`",
                f"- fallback_layers: `{info['fallback_layers']}`",
                f"- p2_changed_decision_count: `{info['p2_changed_decision_count']}`",
                f"- p2_no_effect_count: `{info['p2_no_effect_count']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    repo_root = ROOT
    fixture_dir = Path(args.fixture_dir)
    output_dir = Path(args.output_dir)
    replay = read_json(repo_root / "outputs/offline/m6h_safe_u_closure/replay_suite_summary.json")
    prediction = read_json(repo_root / "outputs/offline/m6p_pre_gpu_fix/prediction_replay_summary.json")

    phase_rows = replay["paired_b_vs_u"]["phase_sync_policy_suite"]["rows"]
    table_c_rows = replay["table_c"]["rows"]
    phase_index = {(row["policy_name"], str(row["layer_id"])): row for row in phase_rows}
    table_c_index = {(row["policy_name"], str(row["layer_id"]), row["p2_source"]): row for row in table_c_rows}
    prediction_rows = prediction["rows"]

    decision_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for safe_policy, family in SAFE_POLICY_FAMILIES.items():
        rows = [row for row in prediction_rows if row["policy_name"] == safe_policy]
        layer_ids = sorted(
            {str(row["target_layer_id"]) for row in rows if str(row.get("target_layer_id", "")).isdigit()},
            key=int,
        )
        benefit_layers: list[str] = []
        harm_layers: list[str] = []
        fallback_layers: list[str] = []
        changed_decisions = 0
        no_effect_count = 0
        for layer_id in layer_ids:
            fixture_path = fixture_dir / f"replay_layer_{layer_id}.json"
            fixture = read_json(fixture_path) if fixture_path.exists() else None
            zero_row = next((row for row in rows if str(row.get("target_layer_id", "")) == layer_id and row["p2_source"] == "zero_hint"), None)
            if zero_row is None:
                continue
            zero_selected = zero_row["selected_policy"]
            for variant in [
                "zero_hint",
                "copy_current_dispatch",
                "fate_style_history",
                "fate_style_linear",
                "perfect_trace_oracle",
                "actual_trace_oracle",
            ]:
                row = next((candidate for candidate in rows if str(candidate.get("target_layer_id", "")) == layer_id and candidate["p2_source"] == variant), None)
                if row is None:
                    continue
                paired_b_row = phase_index.get((family["paired_B"], str(int(layer_id) - 1)))
                raw_u_row = table_c_index.get((family["raw_U"], str(int(layer_id) - 1), variant))
                safe_row = table_c_index.get((safe_policy, str(int(layer_id) - 1), variant))
                p2_matrix = None if fixture is None else _p2_matrix_for_variant(fixture, variant)
                p2_signature = None if p2_matrix is None else flatten_topology_signature(p2_matrix)
                selected_policy = row["selected_policy"]
                fallback = bool(row["fallback_to_B"])
                if selected_policy != zero_selected:
                    changed_decisions += 1
                else:
                    no_effect_count += 1
                if fallback:
                    fallback_layers.append(layer_id)
                if paired_b_row and safe_row:
                    delta = pct_delta(float(paired_b_row["makespan"]), float(safe_row["makespan"]))
                    if delta is not None and delta < 0:
                        benefit_layers.append(layer_id)
                    elif delta is not None and delta > 0:
                        harm_layers.append(layer_id)
                decision_rows.append(
                    {
                        "safe_policy": safe_policy,
                        "heuristic_family": family["family"],
                        "layer_id": layer_id,
                        "window_id": layer_id,
                        "phase_group": "runtime_lookahead",
                        "p2_source": variant,
                        "paired_B_policy": family["paired_B"],
                        "raw_U_policy": family["raw_U"],
                        "paired_B_makespan": None if paired_b_row is None else paired_b_row["makespan"],
                        "raw_U_makespan": None if raw_u_row is None else raw_u_row["makespan"],
                        "safe_U_makespan": None if safe_row is None else safe_row["makespan"],
                        "safe_selected": "B" if fallback else "U",
                        "selected_policy": selected_policy,
                        "fallback_reason": "safe_fallback_to_paired_b" if fallback else "raw_u_selected",
                        "raw_U_delta_vs_B": None if paired_b_row is None or raw_u_row is None else pct_delta(float(paired_b_row["makespan"]), float(raw_u_row["makespan"])),
                        "safe_delta_vs_B": None if paired_b_row is None or safe_row is None else pct_delta(float(paired_b_row["makespan"]), float(safe_row["makespan"])),
                        "p2_remote_bytes": row["forecast_remote_bytes"],
                        "p2_top_edges": None if p2_signature is None else p2_signature["top_edges"],
                        "priority_order_diff_count": None,
                        "priority_order_diff_count_unavailable_reason": "policy ordering explain API is not exported by the current replay path",
                        "critical_edge_changed": None if p2_signature is None else bool(p2_signature["top_edges"]),
                        "bottleneck_src_changed": None if p2_signature is None else p2_signature["bottleneck_src_rank"],
                        "bottleneck_dst_changed": None if p2_signature is None else p2_signature["bottleneck_dst_rank"],
                    }
                )
        summary[safe_policy] = {
            "benefit_layers": sorted(set(benefit_layers), key=int),
            "harm_layers": sorted(set(harm_layers), key=int),
            "fallback_layers": sorted(set(fallback_layers), key=int),
            "p2_changed_decision_count": changed_decisions,
            "p2_no_effect_count": no_effect_count,
            "dominant_bottleneck_patterns": ["see per-layer p2_top_edges and bottleneck ranks"],
        }

    payload = {
        "fixture_dir": str(fixture_dir),
        "decision_rows": decision_rows,
        "summary": summary,
    }
    write_json(output_dir / "safe_u_decision_diff.json", payload)
    write_text(output_dir / "safe_u_decision_diff.md", render_markdown(payload))


if __name__ == "__main__":
    main()
