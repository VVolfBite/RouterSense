#!/usr/bin/env python3
"""Build offline scheduling fixtures from replay traces or phase-context artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.offline.replay_fixture import (
    build_replay_fixture_audit_summary,
    build_replay_fixture_bundle,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _find_trace_paths(trace_dir: Path) -> list[Path]:
    return sorted(path for path in trace_dir.glob("rank*_control_replay_trace.jsonl") if path.is_file())


def _find_phase_context_paths(trace_dir: Path) -> list[Path]:
    return sorted(path for path in trace_dir.glob("rank*_phase_contexts.jsonl") if path.is_file())


def _phase_context_to_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "run_id_digest": str((row.get("plan_key", {}) or {}).get("run_id_digest", "")),
                "layer_id": str(row.get("layer_id", "")),
                "layer_name": str(row.get("layer_name", "")),
                "phase": str(row.get("phase", "")),
                "global_rank": int(row.get("global_rank", 0)),
                "local_rank": int(row.get("local_rank", 0)),
                "ep_group_size": int(len(row.get("per_peer_bytes", []) or [])),
                "policy_name": str(row.get("policy_name", row.get("control_mode", "phase_context_fallback"))),
                "bucket_rows": 0,
                "per_rank_peer_bytes": [int(value) for value in row.get("per_peer_bytes", []) or []],
                "nonzero_edges": [],
                "nonzero_edge_count": int(row.get("nonzero_edge_count", 0) or 0),
            }
        )
    return payload




def _render_audit_summary_md(summary: dict[str, Any]) -> str:
    lines = ["# Replay Fixture Audit Summary", ""]
    if summary["source_kind"] == "phase_context_fallback":
        lines.extend(
            [
                "This fixture was derived from phase contexts rather than control replay trace.",
                "",
            ]
        )
    if int(summary.get("layer_count_with_missing_rank", 0)) > 0:
        lines.extend(
            [
                f"Warning: {summary['layer_count_with_missing_rank']} layer(s) are missing one or more ranks in P0/P1 observations.",
                "",
            ]
        )
    lines.extend(
        [
            "## Global",
            f"- source_kind: `{summary['source_kind']}`",
            f"- trace_file_count: {summary['trace_file_count']}",
            f"- policy_name: `{summary['policy_name']}`",
            f"- run_id_digest: `{summary['run_id_digest']}`",
            f"- layer_count: {summary['layer_count']}",
            f"- fixture_count: {summary['fixture_count']}",
            f"- num_gpus: {summary['num_gpus']}",
            f"- expected_rank_count: {summary['expected_rank_count']}",
            "",
            "## Totals",
            f"- layer_count_with_complete_p0p1: {summary['layer_count_with_complete_p0p1']}",
            f"- layer_count_with_missing_rank: {summary['layer_count_with_missing_rank']}",
            f"- total_p0_bytes: {summary['total_p0_bytes']}",
            f"- total_p1_bytes: {summary['total_p1_bytes']}",
            f"- total_p2_bytes: {summary['total_p2_bytes']}",
            f"- total_p0_self_bytes: {summary.get('total_p0_self_bytes', 0)}",
            f"- total_p1_self_bytes: {summary.get('total_p1_self_bytes', 0)}",
            f"- total_p2_self_bytes: {summary.get('total_p2_self_bytes', 0)}",
            "",
            "## Layers",
            "| Fixture | Layer | Next | P0 Missing | P1 Missing | P0 Bytes | P1 Bytes | P2 Bytes | P2 Source |",
            "|---|---:|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["layers"]:
        lines.append(
            f"| {row['fixture_name']} | {row['layer_id']} | {row['next_layer_id'] or '-'} | "
            f"{row['p0_missing_ranks']} | {row['p1_missing_ranks']} | {row['p0_total_bytes']} | "
            f"{row['p1_total_bytes']} | {row['p2_total_bytes']} | {row['p2_source']} |"
        )
    return "\n".join(lines) + "\n"


def _write_bundle(bundle: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id_digest": bundle.get("run_id_digest", ""),
        "policy_name": bundle.get("policy_name", ""),
        "layer_count": int(bundle.get("layer_count", 0)),
        "fixture_count": int(bundle.get("fixture_count", 0)),
        "fixture_names": [str(item.get("fixture_name", "")) for item in bundle.get("fixtures", [])],
    }
    (output_dir / "replay_fixture_bundle_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "replay_fixture_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fixture_dir = output_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for fixture in bundle.get("fixtures", []):
        payload = {
            "num_gpus": int(fixture["num_gpus"]),
            "p0_dispatch_matrix": fixture["p0_dispatch_matrix"],
            "p1_return_matrix": fixture["p1_return_matrix"],
            "p2_next_dispatch_forecast_matrix": fixture["p2_next_dispatch_forecast_matrix"],
            "p2_next_dispatch_matrix": fixture["p2_next_dispatch_matrix"],
            "metadata": fixture["metadata"],
        }
        (fixture_dir / f"{fixture['fixture_name']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True, help="Directory containing rank*_control_replay_trace.jsonl files")
    parser.add_argument("--policy", default="", help="Optional policy filter")
    parser.add_argument("--output-dir", required=True, help="Directory to write generated fixtures")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trace_dir = Path(args.trace_dir)
    trace_paths = _find_trace_paths(trace_dir)
    rows: list[dict[str, Any]] = []
    source_kind = "control_replay_trace"
    if trace_paths:
        for path in trace_paths:
            rows.extend(_read_jsonl(path))
    else:
        phase_context_paths = _find_phase_context_paths(trace_dir)
        if not phase_context_paths:
            raise SystemExit(
                f"no rank*_control_replay_trace.jsonl or rank*_phase_contexts.jsonl files found under {trace_dir}"
            )
        source_kind = "phase_context_fallback"
        for path in phase_context_paths:
            rows.extend(_phase_context_to_trace_rows(_read_jsonl(path)))
    bundle = build_replay_fixture_bundle(rows, policy_name=(args.policy or None))
    output_dir = Path(args.output_dir)
    _write_bundle(bundle, output_dir)
    audit_summary = build_replay_fixture_audit_summary(
        bundle,
        source_kind=source_kind,
        trace_file_count=len(trace_paths),
    )
    (output_dir / "replay_fixture_audit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "replay_fixture_audit_summary.md").write_text(
        _render_audit_summary_md(audit_summary),
        encoding="utf-8",
    )
    print(json.dumps({
        "trace_dir": str(trace_dir),
        "trace_file_count": len(trace_paths),
        "source_kind": source_kind,
        "fixture_count": bundle["fixture_count"],
        "policy_name": bundle["policy_name"],
        "output_dir": str(Path(args.output_dir)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
