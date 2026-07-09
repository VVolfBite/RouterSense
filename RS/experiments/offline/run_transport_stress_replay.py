#!/usr/bin/env python3
"""Communication-only replay suite over real-trace-derived fixtures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline.run_replay_fixture_policy_suite import TABLE_A_POLICIES, TABLE_B_POLICIES, run_bridge_suite, run_policy_suite


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


def _augment_transport_summary(summary_rows: list[dict[str, Any]], *, total_bytes: int, relative_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        mean_makespan = row.get("mean_makespan")
        comm_throughput = None
        if mean_makespan not in (None, 0.0):
            comm_throughput = float(total_bytes / float(mean_makespan))
        rows.append(
            {
                **row,
                "communication_only": True,
                "transport_total_bytes": int(total_bytes),
                "normalized_byte_throughput": comm_throughput,
                "improvement_vs_baseline_pct": (
                    None
                    if row.get(relative_key) is None
                    else float(-100.0 * float(row[relative_key]))
                ),
            }
        )
    return rows


def _render_md(payload: dict[str, Any]) -> str:
    audit = payload["fixture_audit"]
    phase_sync = payload["phase_sync_transport"]["summary"]
    joint = payload["joint_transport"]["summary"]
    lines = [
        "# Transport Stress Replay Summary",
        "",
        "这份报告只看由真实 online trace 还原出的通信窗口，不包含完整 forward compute。",
        "",
        "## Fixture source",
        "",
        f"- fixture_dir: `{payload['fixture_dir']}`",
        f"- source_kind: `{audit['source_kind']}`",
        f"- layer_count: {audit['layer_count']}",
        f"- fixture_count: {audit['fixture_count']}",
        f"- total_p0_bytes: {audit['total_p0_bytes']}",
        f"- total_p1_bytes: {audit['total_p1_bytes']}",
        f"- total_p2_bytes: {audit['total_p2_bytes']}",
        "",
        "## Phase-sync-compatible transport replay",
        "",
        "| Policy | Mean Comm Makespan | Mean Waves | Relative to Birkhoff | Throughput(bytes/unit) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in phase_sync:
        rel = row["relative_to_birkhoff_phase_local"]
        lines.append(
            f"| {row['policy_name']} | {row['mean_makespan']:.0f} | {row['mean_wave_count']:.2f} | "
            f"{('-' if rel is None else f'{rel * 100:.2f}%')} | "
            f"{('-' if row['normalized_byte_throughput'] is None else f'{row['normalized_byte_throughput']:.6f}')} |"
        )
    lines.extend(
        [
            "",
            "## Joint upper-bound transport replay",
            "",
            "| Policy | Mean Comm Makespan | Mean Waves | Relative to B_birkhoff_wave | Throughput(bytes/unit) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in joint:
        rel = row["relative_to_B_birkhoff_wave"]
        lines.append(
            f"| {row['policy_name']} | {row['mean_makespan']:.0f} | {row['mean_wave_count']:.2f} | "
            f"{('-' if rel is None else f'{rel * 100:.2f}%')} | "
            f"{('-' if row['normalized_byte_throughput'] is None else f'{row['normalized_byte_throughput']:.6f}')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 这是 communication-only logical replay，不是 GPU runtime latency。",
            "- phase-sync-compatible 表对应当前 online 保守语义。",
            "- joint upper-bound 表对应 offline execution-window 语义。",
            "- 如果 joint upper-bound 明显优于 phase-local baseline，说明通信段本身仍有可挖掘空间。",
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
    joint = run_policy_suite(
        fixture_dir=suite_fixture_dir,
        policies=TABLE_B_POLICIES,
        mode="execution_window",
        p2_source="actual_trace",
        expert_compute_delay=float(args.expert_compute_delay),
        baseline_policy="B_birkhoff_wave",
        relative_key="relative_to_B_birkhoff_wave",
    )
    phase_sync_rows = _augment_transport_summary(
        phase_sync["summary"],
        total_bytes=int(audit.get("total_p0_bytes", 0)) + int(audit.get("total_p1_bytes", 0)),
        relative_key="relative_to_birkhoff_phase_local",
    )
    joint_rows = _augment_transport_summary(
        joint["summary"],
        total_bytes=int(audit.get("total_p0_bytes", 0)) + int(audit.get("total_p1_bytes", 0)) + int(audit.get("total_p2_bytes", 0)),
        relative_key="relative_to_B_birkhoff_wave",
    )
    bridge = run_bridge_suite(
        fixture_dir=suite_fixture_dir,
        expert_compute_delay=float(args.expert_compute_delay),
    )
    payload = {
        "fixture_dir": str(suite_fixture_dir),
        "fixture_audit": audit,
        "phase_sync_transport": {
            **phase_sync,
            "summary": phase_sync_rows,
            "mean_transport_makespan": statistics.mean([row["mean_makespan"] for row in phase_sync_rows if row["mean_makespan"] is not None]),
        },
        "joint_transport": {
            **joint,
            "summary": joint_rows,
            "mean_transport_makespan": statistics.mean([row["mean_makespan"] for row in joint_rows if row["mean_makespan"] is not None]),
        },
        "bridge_candidates": bridge,
    }
    (output_dir / "transport_stress_phase_sync_summary.json").write_text(
        json.dumps(payload["phase_sync_transport"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "transport_stress_joint_summary.json").write_text(
        json.dumps(payload["joint_transport"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "transport_stress_replay_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "transport_stress_bridge_summary.json").write_text(
        json.dumps(bridge, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "transport_stress_replay_summary.md").write_text(
        _render_md(payload),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "fixture_dir": str(suite_fixture_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
