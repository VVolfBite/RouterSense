#!/usr/bin/env python3
"""Generate offline evidence tables from replay-derived real-trace fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline.run_replay_fixture_policy_suite import (
    TABLE_A_POLICIES,
    TABLE_B_POLICIES,
    build_oracle_table,
    run_paired_suite,
    run_bridge_suite,
    run_policy_suite,
    run_prediction_u_suite,
    summarize_best_pair,
    summarize_best_u_frontier,
)
from rs.scheduling.algorithm_catalog import pair_status_summary
from rs.scheduling.algorithm_catalog import pair_status_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", default="")
    parser.add_argument("--bundle", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--strict-audit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _fixture_dir_from_args(args: argparse.Namespace) -> Path:
    if args.fixture_dir:
        return Path(args.fixture_dir)
    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        raise SystemExit("either --fixture-dir or --bundle must point to an existing replay fixture bundle")
    return bundle_path.parent / "fixtures"


def _load_audit(fixture_dir: Path) -> dict[str, Any]:
    audit_path = fixture_dir.parent / "replay_fixture_audit_summary.json"
    if not audit_path.exists():
        raise SystemExit(f"missing fixture audit summary: {audit_path}")
    return json.loads(audit_path.read_text(encoding="utf-8"))


def _limit_fixture_dir(src_dir: Path, dst_dir: Path, max_layers: int) -> Path:
    if max_layers <= 0:
        return src_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(sorted(src_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))):
        if index >= max_layers:
            break
        (dst_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return dst_dir


def _render_md(payload: dict[str, Any]) -> str:
    audit = payload["fixture_audit"]
    lines = [
        "# Real Trace Evidence Summary",
        "",
        "## Fixture audit",
        "",
        f"- fixture_dir: `{payload['fixture_dir']}`",
        f"- source_kind: `{audit['source_kind']}`",
        f"- layer_count: {audit['layer_count']}",
        f"- fixture_count: {audit['fixture_count']}",
        f"- layer_count_with_missing_rank: {audit['layer_count_with_missing_rank']}",
        f"- total_p0_bytes: {audit['total_p0_bytes']}",
        f"- total_p1_bytes: {audit['total_p1_bytes']}",
        f"- total_p2_bytes: {audit['total_p2_bytes']}",
        f"- ready_pair_count: {payload['pair_status']['ready_pair_count']}",
        f"- pending_pair_count: {payload['pair_status']['pending_pair_count']}",
        "- P2 uses next-layer actual P0 when available; the last layer uses a zero matrix.",
        "",
        "## Paired B-vs-U result",
        "",
        "| Family | B | Safe U | B Mean | Safe U Mean | Safe U vs B | Ready |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in payload["paired_b_vs_u"]["summary"]:
        improvement = row["safe_U_vs_B_improvement_pct"]
        b_mean = "-" if row["B_mean_makespan"] is None else f"{float(row['B_mean_makespan']):.0f}"
        u_mean = "-" if row["safe_U_mean_makespan"] is None else f"{float(row['safe_U_mean_makespan']):.0f}"
        improvement_text = "-" if improvement is None else f"{float(improvement):.2f}%"
        lines.append(
            f"| {row['heuristic_family']} | {row['B_algorithm']} | {row['safe_U_algorithm']} | "
            f"{b_mean} | "
            f"{u_mean} | "
            f"{improvement_text} | "
            f"{row['paired_comparison_ready']} |"
        )
    lines.extend(
        [
            "",
            "多个 pair 中如果 U 系统性优于 B，说明收益来自联合调度建模，而不是来自换了完全不同的启发式。",
            f"- best_pair_family: `{payload['best_pair']['best_family']}`",
            f"- best_pair_improvement_pct: {payload['best_pair']['best_improvement_pct']}",
            "Pending pair reasons:",
        ]
    )
    for row in payload["pair_status"]["pending_pairs"]:
        lines.append(f"- `{row['heuristic_family']}`: {row['pending_reason']}")
    lines.append("")
    lines.append("## Joint scheduling space")
    lines.append("")
    lines.append("| Policy | Valid | Mean Makespan | Relative to B_birkhoff_wave |")
    lines.append("|---|---:|---:|---:|")
    for row in payload["execution_window_joint"]["summary"]:
        rel = row["relative_to_B_birkhoff_wave"]
        lines.append(
            f"| {row['policy_name']} | {row['valid_layer_count']} | {row['mean_makespan']:.0f} | "
            f"{('-' if rel is None else f'{rel * 100:.2f}%')} |"
        )
    lines.extend(
        [
            "",
            "这张表只对应 offline execution-window / oracle-information joint scheduling study，不能被解读为当前 online RouterSense 已实现收益。",
            f"- best_execution_window_u_algorithm: `{payload['best_u_frontier']['best_execution_window_u_algorithm']}`",
            f"- best_execution_window_gap_to_B_birkhoff_wave_pct: {payload['best_u_frontier']['best_execution_window_gap_to_B_birkhoff_wave_pct']}",
            "",
            "## Cross-layer prediction value",
            "",
            "| U Policy | Family | P2 Source | Mean Makespan | Relative to zero_hint | Relative to perfect_trace |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in payload["prediction_oracle"]["summary"]:
        rel_zero = row["relative_to_zero_hint"]
        rel_perfect = row["relative_to_perfect_trace"]
        mean_text = "-" if row["mean_makespan"] is None else f"{float(row['mean_makespan']):.0f}"
        lines.append(
            f"| {row['U_algorithm']} | {row['heuristic_family']} | {row['p2_source']} | {mean_text} | "
            f"{('-' if rel_zero is None else f'{rel_zero * 100:.2f}%')} | "
            f"{('-' if rel_perfect is None else f'{rel_perfect * 100:.2f}%')} | "
            f"{row['future_information_mode']} |"
        )
    lines.extend(
        [
            "",
            "zero_hint 表示没有跨层预测；copy_current_dispatch 是廉价启发式；perfect_trace_oracle / actual_trace_oracle 仅代表 oracle predict 上界，不是实时 predictor。",
            "",
            "## Oracle table",
            "",
            "| Oracle | Type | Implementation | Objective | Heavy Solver |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["oracle_table"]["summary"]:
        lines.append(
            f"| {row['oracle_name']} | {row['oracle_type']} | {row['implementation']} | "
            f"{row['objective']} | {row['heavy_solver']} |"
        )
    lines.extend(
        [
            "",
            "## RouterSense bridge candidates",
            "",
            "| Policy | Mean Makespan | Rel to Birkhoff | Rel to Current RouterSense | Gap to Best U | Eval Mode |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["bridge_candidates"]["summary"]:
        rel_b = row.get("relative_to_birkhoff_phase_local")
        rel_r = row.get("relative_to_current_routersense")
        gap_u = row.get("gap_to_best_joint_raw_u")
        mean = row.get("mean_makespan")
        lines.append(
            f"| {row['policy_name']} | {('-' if mean is None else f'{float(mean):.0f}')} | "
            f"{('-' if rel_b is None else f'{float(rel_b) * 100:.2f}%')} | "
            f"{('-' if rel_r is None else f'{float(rel_r) * 100:.2f}%')} | "
            f"{('-' if gap_u is None else f'{float(gap_u) * 100:.2f}%')} | "
            f"{row.get('evaluation_mode', '')} |"
        )
    lines.extend(
        [
            "",
            "## Online runtime interpretation",
            "",
            "- 当前 phase_sync online 是真实可执行的保守线。",
            "- current RouterSense hint policy 不是 full joint execution-window scheduler。",
            "- async_release 目前只有 shadow-only skeleton，还没有 executor integration。",
            "- prepared-plan 的 P2 矩阵在真实分布式 EP group 可用时可以来自 gathered_global_matrix；但这只是修正全局矩阵来源，不等于真实 next-layer predictor 已接入。",
            "- 下一步需要 bridge policy + async_release executor integration，才能把 offline joint 空间转换成在线系统收益。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    fixture_dir = _fixture_dir_from_args(args)
    audit = _load_audit(fixture_dir)
    if args.strict_audit and int(audit.get("layer_count_with_missing_rank", 0)) > 0:
        raise SystemExit("fixture audit reports missing ranks; rerun with --no-strict-audit to continue")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_fixture_dir = fixture_dir
    if int(args.max_layers) > 0:
        suite_fixture_dir = _limit_fixture_dir(fixture_dir, output_dir / "fixtures_subset", int(args.max_layers))
    phase_sync = run_policy_suite(
        fixture_dir=suite_fixture_dir,
        policies=TABLE_A_POLICIES,
        mode="runtime_lookahead",
        p2_source="copy_current_dispatch",
        expert_compute_delay=float(args.expert_compute_delay),
        baseline_policy="birkhoff_phase_local",
        relative_key="relative_to_birkhoff_phase_local",
    )
    execution_window = run_policy_suite(
        fixture_dir=suite_fixture_dir,
        policies=TABLE_B_POLICIES,
        mode="execution_window",
        p2_source="actual_trace",
        expert_compute_delay=float(args.expert_compute_delay),
        baseline_policy="B_birkhoff_wave",
        relative_key="relative_to_B_birkhoff_wave",
    )
    prediction = run_prediction_u_suite(
        fixture_dir=suite_fixture_dir,
        p2_sources=("zero_hint", "copy_current_dispatch", "fate_style_history", "fate_style_linear", "perfect_trace", "actual_trace"),
        expert_compute_delay=float(args.expert_compute_delay),
    )
    paired = run_paired_suite(
        fixture_dir=suite_fixture_dir,
        expert_compute_delay=float(args.expert_compute_delay),
    )
    oracle_table = build_oracle_table()
    pair_status = pair_status_summary()
    best_pair = summarize_best_pair(paired)
    best_u_frontier = summarize_best_u_frontier(
        paired_summary=paired,
        execution_window_summary=execution_window,
    )
    bridge = run_bridge_suite(
        fixture_dir=suite_fixture_dir,
        expert_compute_delay=float(args.expert_compute_delay),
    )
    payload = {
        "fixture_dir": str(suite_fixture_dir),
        "fixture_audit": audit,
        "phase_sync_compatible": phase_sync,
        "execution_window_joint": execution_window,
        "prediction_oracle": prediction,
        "paired_b_vs_u": paired,
        "oracle_table": oracle_table,
        "pair_status": pair_status,
        "best_pair": best_pair,
        "best_u_frontier": best_u_frontier,
        "bridge_candidates": bridge,
    }
    (output_dir / "phase_sync_compatible_summary.json").write_text(
        json.dumps(phase_sync, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "execution_window_joint_summary.json").write_text(
        json.dumps(execution_window, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "prediction_oracle_summary.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "paired_b_vs_u_summary.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "oracle_table_summary.json").write_text(
        json.dumps(oracle_table, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "bridge_candidates_summary.json").write_text(
        json.dumps(bridge, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "real_trace_evidence_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "real_trace_evidence_summary.md").write_text(
        _render_md(payload),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "fixture_dir": str(suite_fixture_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
