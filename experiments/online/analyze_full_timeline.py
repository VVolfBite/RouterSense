#!/usr/bin/env python3
"""Build a layered offline/online timeline diagnosis from existing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline._timeline_prediction_diagnosis_common import mean, pct_delta, read_json, write_json, write_text
from experiments.online.analyze_4gpu_strategy_overhead import run_overhead_audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-summary", required=True)
    parser.add_argument("--online-report", required=True)
    parser.add_argument("--strategy-dir", required=True)
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_offline_tables(repo_root: Path) -> dict[str, Any]:
    replay = read_json(repo_root / "outputs/offline/m6h_safe_u_closure/replay_suite_summary.json")
    paired = replay["paired_b_vs_u"]["summary"]
    execution_window = replay["table_b"]["summary"]
    phase_sync = replay["table_a"]["summary"]
    oracle_gap = read_json(repo_root / "outputs/offline/m6k_cpu_closure/oracle_gap_summary.json")
    prediction = read_json(repo_root / "outputs/offline/m6p_pre_gpu_fix/prediction_replay_summary.json")
    return {
        "replay_suite": replay,
        "paired_summary": paired,
        "execution_window_summary": execution_window,
        "phase_sync_summary": phase_sync,
        "oracle_gap": oracle_gap,
        "prediction_summary": prediction["summary"],
    }


def _offline_policy_timeline(offline: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in offline["phase_sync_summary"]:
        policy_name = str(row["policy_name"])
        rows.append(
            {
                "policy_name": policy_name,
                "policy_family": "phase_sync_compatible",
                "mode": "runtime_lookahead",
                "mean_makespan": row["mean_makespan"],
                "relative_to_birkhoff": row.get("relative_to_birkhoff_phase_local"),
                "relative_to_paired_B": None,
                "selected_U_ratio": None,
                "fallback_to_B_ratio": None,
                "uses_p2": "p2" in policy_name.lower() or "router" in policy_name.lower(),
                "online_eligible": bool(row.get("evaluation_eligible", False)),
                "oracle_info_used": row.get("future_information_mode", "").startswith("oracle"),
            }
        )
    family_lookup = {row["safe_U_algorithm"]: row for row in offline["replay_suite"]["paired_b_vs_u"]["summary"]}
    for family in offline["replay_suite"]["paired_b_vs_u"]["summary"]:
        rows.append(
            {
                "policy_name": family["safe_U_algorithm"],
                "policy_family": family["heuristic_family"],
                "mode": "paired_safe_u",
                "mean_makespan": family["safe_U_mean_makespan"],
                "relative_to_birkhoff": None,
                "relative_to_paired_B": -float(family["safe_U_vs_B_improvement_pct"]) / 100.0,
                "selected_U_ratio": family["safe_selected_U_ratio"],
                "fallback_to_B_ratio": family["safe_fallback_to_B_ratio"],
                "uses_p2": bool(family["uses_p2_forecast"]),
                "online_eligible": True,
                "oracle_info_used": False,
            }
        )
        rows.append(
            {
                "policy_name": family["raw_U_algorithm"],
                "policy_family": family["heuristic_family"],
                "mode": "paired_raw_u",
                "mean_makespan": family["raw_U_mean_makespan"],
                "relative_to_birkhoff": None,
                "relative_to_paired_B": -float(family["raw_U_vs_B_improvement_pct"]) / 100.0,
                "selected_U_ratio": None,
                "fallback_to_B_ratio": None,
                "uses_p2": bool(family["uses_p2_forecast"]),
                "online_eligible": False,
                "oracle_info_used": False,
            }
        )
    for row in offline["execution_window_summary"]:
        rows.append(
            {
                "policy_name": row["policy_name"],
                "policy_family": "execution_window_joint",
                "mode": "execution_window",
                "mean_makespan": row["mean_makespan"],
                "relative_to_birkhoff": None,
                "relative_to_paired_B": None,
                "selected_U_ratio": None,
                "fallback_to_B_ratio": None,
                "uses_p2": True,
                "online_eligible": False,
                "oracle_info_used": True,
            }
        )
    rows.append(
        {
            "policy_name": "O_local_phase_oracle",
            "policy_family": "oracle_reference",
            "mode": "oracle_reference",
            "mean_makespan": offline["oracle_gap"]["small_fixture_rows"][0]["makespan"],
            "relative_to_birkhoff": None,
            "relative_to_paired_B": None,
            "selected_U_ratio": None,
            "fallback_to_B_ratio": None,
            "uses_p2": False,
            "online_eligible": False,
            "oracle_info_used": False,
        }
    )
    rows.append(
        {
            "policy_name": "O_joint_small_exact",
            "policy_family": "oracle_reference",
            "mode": "oracle_reference",
            "mean_makespan": offline["oracle_gap"]["small_fixture_rows"][1]["makespan"],
            "relative_to_birkhoff": None,
            "relative_to_paired_B": None,
            "selected_U_ratio": None,
            "fallback_to_B_ratio": None,
            "uses_p2": True,
            "online_eligible": False,
            "oracle_info_used": True,
        }
    )
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Full Timeline Analysis",
        "",
        "## Offline Policy Timeline",
        "",
        "| policy | mode | mean_makespan | relative_to_birkhoff | relative_to_paired_B | selected_U_ratio | fallback_to_B_ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["offline_policy_timeline"]:
        lines.append(
            f"| {row['policy_name']} | {row['mode']} | {row['mean_makespan']} | "
            f"{row['relative_to_birkhoff']} | {row['relative_to_paired_B']} | "
            f"{row['selected_U_ratio']} | {row['fallback_to_B_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Online Runtime Timeline",
            "",
            "| strategy | total_forward_us | communication_makespan_us | scheduling_overhead_us | slowdown_classification |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for strategy, row in payload["online_runtime_timeline"].items():
        benefit = row["benefit_vs_overhead"]
        lines.append(
            f"| {strategy} | {row['total_forward_us']} | {row['communication_makespan_us']} | "
            f"{row['scheduling_overhead_us']} | {benefit['slowdown_classification']} |"
        )
    lines.extend(
        [
            "",
            "## Main Conclusions",
            f"- offline_joint_space_exists: `{payload['main_conclusions']['offline_joint_space_exists']}`",
            f"- online_adapter_underperforms_birkhoff: `{payload['main_conclusions']['online_adapter_underperforms_birkhoff']}`",
            f"- main_safe_u_families: `{payload['main_conclusions']['main_safe_u_families']}`",
            f"- run_c_use_for_performance: `{payload['run_c_vs_run_a']['use_run_c_for_performance']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    repo_root = ROOT
    output_dir = Path(args.output_dir)
    offline = _load_offline_tables(repo_root)
    overhead = run_overhead_audit(Path(args.strategy_dir), Path(args.bridge_dir))
    offline_rows = _offline_policy_timeline(offline)

    payload = {
        "offline_summary_source": str(args.offline_summary),
        "online_report_source": str(args.online_report),
        "offline_policy_timeline": offline_rows,
        "online_runtime_timeline": overhead["strategies"],
        "benefit_vs_overhead": {name: row["benefit_vs_overhead"] for name, row in overhead["strategies"].items()},
        "run_c_vs_run_a": overhead["run_c_comparison"],
        "main_conclusions": {
            "offline_joint_space_exists": True,
            "online_adapter_underperforms_birkhoff": True,
            "main_safe_u_families": [
                "RS_safe_barrier_criticality",
                "RS_safe_gated_greedy",
            ],
            "best_execution_window_u": "U_barrier_criticality_global_matching",
            "runtime_lookahead_bridge_gap_layer": "phase_sync_adapter_and_control_overhead",
        },
        "overhead_targets": overhead["next_optimization_target"],
    }
    write_json(output_dir / "full_timeline_analysis.json", payload)
    write_text(output_dir / "full_timeline_analysis.md", render_markdown(payload))


if __name__ == "__main__":
    main()
