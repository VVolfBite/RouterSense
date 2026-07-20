#!/usr/bin/env python3
"""Inventory real 4GPU artifacts before deeper diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline._timeline_prediction_diagnosis_common import read_json, read_jsonl, write_json, write_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-dir", required=True)
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _count_rank_artifacts(run_dir: Path, patterns: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for pattern in patterns:
        files = sorted(run_dir.glob(pattern))
        payload[pattern] = {
            "file_count": len(files),
            "files": [str(path) for path in files[:16]],
            "non_empty_file_count": sum(1 for path in files if path.exists() and path.stat().st_size > 0),
        }
    return payload


def _rank_layer_stats(run_dir: Path, pattern: str) -> dict[str, Any]:
    layer_ids: set[str] = set()
    phase_ids: set[str] = set()
    rank_counts: dict[str, int] = {}
    for path in sorted(run_dir.glob(pattern)):
        rank_name = path.name.split("_", 1)[0]
        rows = read_jsonl(path)
        rank_counts[rank_name] = len(rows)
        for row in rows:
            if "layer_id" in row:
                layer_ids.add(str(row["layer_id"]))
            if "phase" in row:
                phase_ids.add(str(row["phase"]))
    return {
        "rank_counts": rank_counts,
        "layer_ids": sorted(layer_ids, key=lambda value: int(value) if value.isdigit() else value),
        "layer_count": len(layer_ids),
        "phase_ids": sorted(phase_ids),
        "phase_count": len(phase_ids),
    }


def _collect_strategy_runs(base_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    required = [
        "rank*_transport_execution.jsonl",
        "rank*_scheduled_phase_plans.jsonl",
        "rank*_phase_contexts.jsonl",
        "rank*_control_timeline.jsonl",
        "rank*_planning_timing.jsonl",
        "rank*_prediction_audit.jsonl",
    ]
    for strategy_dir in sorted(base_dir.iterdir()):
        rep_dir = strategy_dir / "rep0"
        if not rep_dir.is_dir():
            continue
        summary_path = rep_dir / "summary.json"
        manifest_path = rep_dir / "run_manifest.json"
        payload[strategy_dir.name] = {
            "rep_dir": str(rep_dir),
            "summary_exists": summary_path.exists(),
            "manifest_exists": manifest_path.exists(),
            "artifact_counts": _count_rank_artifacts(rep_dir, required),
            "phase_context_stats": _rank_layer_stats(rep_dir, "rank*_phase_contexts.jsonl"),
            "transport_stats": _rank_layer_stats(rep_dir, "rank*_transport_execution.jsonl"),
            "prediction_audit_stats": _rank_layer_stats(rep_dir, "rank*_prediction_audit.jsonl"),
            "summary": read_json(summary_path) if summary_path.exists() else None,
            "manifest": read_json(manifest_path) if manifest_path.exists() else None,
        }
    return payload


def _collect_trace_run(trace_dir: Path) -> dict[str, Any]:
    patterns = [
        "rank*_expert_route_trace.jsonl",
        "rank*_source_expert_counts.jsonl",
        "rank*_expert_to_traffic_audit.jsonl",
        "rank*_expert_trace_warnings.jsonl",
        "rank*_transport_execution.jsonl",
        "rank*_prediction_audit.jsonl",
    ]
    return {
        "trace_dir": str(trace_dir),
        "artifact_counts": _count_rank_artifacts(trace_dir, patterns),
        "source_expert_stats": _rank_layer_stats(trace_dir, "rank*_source_expert_counts.jsonl"),
        "route_trace_stats": _rank_layer_stats(trace_dir, "rank*_expert_route_trace.jsonl"),
        "prediction_audit_stats": _rank_layer_stats(trace_dir, "rank*_prediction_audit.jsonl"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 4GPU Artifact Inventory",
        "",
        f"- strategy_dir: `{payload['strategy_dir']}`",
        f"- bridge_dir: `{payload['bridge_dir']}`",
        f"- trace_dir: `{payload['trace_dir']}`",
        "",
        "## Strategy Runs",
    ]
    for name, info in payload["strategy_runs"].items():
        lines.extend(
            [
                f"### {name}",
                f"- summary_exists: `{info['summary_exists']}`",
                f"- manifest_exists: `{info['manifest_exists']}`",
                f"- phase_context_layers: `{info['phase_context_stats']['layer_count']}`",
                f"- transport_rows_by_rank: `{info['transport_stats']['rank_counts']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Trace Run",
            f"- source_expert_layers: `{payload['trace_run']['source_expert_stats']['layer_count']}`",
            f"- source_expert_rows_by_rank: `{payload['trace_run']['source_expert_stats']['rank_counts']}`",
            "",
            "## Comparability",
            f"- run_a_run_b_run_c_comparable: `{payload['comparability']['comparable']}`",
            f"- differences: `{payload['comparability']['differences']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    strategy_dir = Path(args.strategy_dir)
    bridge_dir = Path(args.bridge_dir)
    trace_dir = Path(args.trace_dir)
    report = read_json(Path(args.report))
    output_dir = Path(args.output_dir)

    strategy_runs = _collect_strategy_runs(strategy_dir)
    bridge_runs = _collect_strategy_runs(bridge_dir)
    trace_run = _collect_trace_run(trace_dir)
    comparability = {
        "comparable": False,
        "differences": [
            "run_b uses debug expert trace collection",
            "run_c is a bridge probe, not a comparison harness",
            "run_a is the only online strategy comparison run for performance interpretation",
        ],
    }
    payload = {
        "strategy_dir": str(strategy_dir),
        "bridge_dir": str(bridge_dir),
        "trace_dir": str(trace_dir),
        "report_path": str(args.report),
        "strategy_runs": strategy_runs,
        "bridge_runs": bridge_runs,
        "trace_run": trace_run,
        "comparability": comparability,
        "collection_report": report,
    }
    write_json(output_dir / "inventory.json", payload)
    write_text(output_dir / "inventory.md", render_markdown(payload))


if __name__ == "__main__":
    main()

