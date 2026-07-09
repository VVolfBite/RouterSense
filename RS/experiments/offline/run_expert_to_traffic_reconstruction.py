#!/usr/bin/env python3
"""Audit expert->traffic reconstruction from expert-route traces when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.online.megatron_ep.prediction.expert_to_traffic import compare_reconstructed_traffic, source_expert_counts_to_traffic_matrix
from rs.runtime.online.megatron_ep.prediction.expert_trace import load_source_expert_counts_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    parser.add_argument("--bytes-per-token", type=int, default=1)
    return parser.parse_args()


def run_expert_to_traffic_reconstruction(*, fixture_dir: Path, bytes_per_token: int = 1) -> dict[str, Any]:
    source_count_files = sorted(fixture_dir.glob("*source_expert_counts*.jsonl"))
    if not source_count_files:
        return {
            "fixture_dir": str(fixture_dir),
            "expert_trace_available": False,
            "expert_trace_unavailable_for_real_fixture": True,
            "gpu_collection_required": True,
            "rows": [],
            "summary": {
                "record_count": 0,
                "mean_relative_l1_error": None,
            },
        }
    rows: list[dict[str, Any]] = []
    for count_path in source_count_files:
        for source_counts in load_source_expert_counts_jsonl(count_path):
            fixture_path = fixture_dir / f"replay_layer_{source_counts.layer_id}.json"
            if not fixture_path.exists():
                continue
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            num_experts = int(source_counts.num_experts)
            world_size = int(source_counts.world_size)
            expert_to_rank = {expert_id: expert_id % max(1, world_size) for expert_id in range(num_experts)}
            reconstructed = source_expert_counts_to_traffic_matrix(
                source_counts,
                expert_to_rank,
                bytes_per_token=bytes_per_token,
            )
            actual = tuple(tuple(int(v) for v in row) for row in fixture["p0_dispatch_matrix"])
            audit = compare_reconstructed_traffic(reconstructed, actual)
            rows.append(
                {
                    "layer_id": source_counts.layer_id,
                    "relative_l1_error": audit.relative_l1_error,
                    "cosine_similarity": audit.cosine_similarity,
                    "topk_edge_overlap": audit.topk_edge_overlap,
                    "self_bytes_ignored": audit.self_bytes_ignored,
                }
            )
    mean_relative = None if not rows else sum(float(row["relative_l1_error"]) for row in rows) / len(rows)
    return {
        "fixture_dir": str(fixture_dir),
        "expert_trace_available": True,
        "gpu_collection_required": False,
        "rows": rows,
        "summary": {
            "record_count": len(rows),
            "mean_relative_l1_error": mean_relative,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Expert To Traffic Reconstruction", ""]
    lines.append(f"- fixture_dir: `{payload['fixture_dir']}`")
    lines.append(f"- expert_trace_available: `{payload['expert_trace_available']}`")
    if not payload["expert_trace_available"]:
        lines.append("- GPU collection required: collect selected_experts, routing_weights, source_rank, layer_id, expert_to_rank")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "| Layer | relative L1 | cosine | top-k overlap | self bytes ignored |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            f"| {row['layer_id']} | {row['relative_l1_error']:.4f} | {row['cosine_similarity']:.4f} | {row['topk_edge_overlap']:.4f} | {row['self_bytes_ignored']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    payload = run_expert_to_traffic_reconstruction(
        fixture_dir=Path(args.fixture_dir),
        bytes_per_token=int(args.bytes_per_token),
    )
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_summary_md).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
