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
from rs.runtime.online.megatron_ep.prediction.expert_trace import SourceExpertCountMatrix, load_source_expert_counts_jsonl


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
            "required_gpu_fields": [
                "selected_experts",
                "routing_weights",
                "source_rank",
                "layer_id",
                "expert_to_rank_map",
                "source_expert_counts",
                "bytes_per_token",
            ],
            "rows": [],
            "summary": {
                "record_count": 0,
                "mean_relative_l1_error": None,
                "expert_to_traffic_mapping_valid": False,
                "source_rank_granularity_required": None,
            },
        }
    rows: list[dict[str, Any]] = []
    o1_errors: list[float] = []
    o2_errors: list[float] = []
    o3_errors: list[float] = []
    o4_errors: list[float] = []
    for count_path in source_count_files:
        for source_counts in load_source_expert_counts_jsonl(count_path):
            fixture_path = fixture_dir / f"replay_layer_{source_counts.layer_id}.json"
            if not fixture_path.exists():
                continue
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            num_experts = int(source_counts.num_experts)
            world_size = int(source_counts.world_size)
            expert_to_rank_map = tuple(
                source_counts.expert_to_rank_map
                or tuple(expert_id % max(1, world_size) for expert_id in range(num_experts))
            )
            expert_to_rank = {expert_id: int(expert_to_rank_map[expert_id]) for expert_id in range(num_experts)}
            reconstructed = source_expert_counts_to_traffic_matrix(
                source_counts,
                expert_to_rank,
                bytes_per_token=int(source_counts.bytes_per_token or bytes_per_token),
            )
            actual = tuple(tuple(int(v) for v in row) for row in fixture["p0_dispatch_matrix"])
            audit = compare_reconstructed_traffic(reconstructed, actual)
            o1_errors.append(float(audit.relative_l1_error))
            global_counts = SourceExpertCountMatrix(
                layer_id=int(source_counts.layer_id),
                world_size=world_size,
                num_experts=num_experts,
                counts=tuple(
                    tuple(
                        int(sum(source_counts.counts[src_rank][expert_id] for src_rank in range(world_size)))
                        if source_rank == 0
                        else 0
                        for expert_id in range(num_experts)
                    )
                    for source_rank in range(world_size)
                ),
                bytes_per_token=int(source_counts.bytes_per_token or bytes_per_token),
                selected_experts_available=bool(source_counts.selected_experts_available),
                routing_weights_available=bool(source_counts.routing_weights_available),
            )
            global_reconstructed = source_expert_counts_to_traffic_matrix(
                global_counts,
                expert_to_rank,
                bytes_per_token=int(source_counts.bytes_per_token or bytes_per_token),
            )
            global_audit = compare_reconstructed_traffic(global_reconstructed, actual)
            o2_errors.append(float(global_audit.relative_l1_error))
            current_expert_copy_audit = None
            current_traffic_copy_audit = None
            next_fixture_path = fixture_dir / f"replay_layer_{int(source_counts.layer_id) + 1}.json"
            if next_fixture_path.exists():
                next_fixture = json.loads(next_fixture_path.read_text(encoding="utf-8"))
                next_actual = tuple(tuple(int(v) for v in row) for row in next_fixture["p0_dispatch_matrix"])
                current_expert_copy_audit = compare_reconstructed_traffic(reconstructed, next_actual)
                current_traffic_copy_audit = compare_reconstructed_traffic(actual, next_actual)
                o3_errors.append(float(current_expert_copy_audit.relative_l1_error))
                o4_errors.append(float(current_traffic_copy_audit.relative_l1_error))
            rows.append(
                {
                    "layer_id": source_counts.layer_id,
                    "actual_source_expert_to_traffic_relative_l1": audit.relative_l1_error,
                    "traffic_cosine": audit.cosine_similarity,
                    "topk_edge_overlap": audit.topk_edge_overlap,
                    "row_sum_error": audit.row_sum_error,
                    "col_sum_error": audit.col_sum_error,
                    "bottleneck_src_match": None,
                    "bottleneck_dst_match": None,
                    "global_expert_count_to_traffic_relative_l1": global_audit.relative_l1_error,
                    "expert_count_copy_baseline_l1": None if current_expert_copy_audit is None else current_expert_copy_audit.relative_l1_error,
                    "traffic_copy_baseline_l1": None if current_traffic_copy_audit is None else current_traffic_copy_audit.relative_l1_error,
                    "self_bytes_ignored": audit.self_bytes_ignored,
                }
            )
    mean_relative = None if not rows else sum(float(row["actual_source_expert_to_traffic_relative_l1"]) for row in rows) / len(rows)
    return {
        "fixture_dir": str(fixture_dir),
        "expert_trace_available": True,
        "gpu_collection_required": False,
        "rows": rows,
        "summary": {
            "record_count": len(rows),
            "mean_relative_l1_error": mean_relative,
            "expert_to_traffic_mapping_valid": mean_relative is not None,
            "source_rank_granularity_required": None if not o1_errors or not o2_errors else (sum(o1_errors) / len(o1_errors)) <= (sum(o2_errors) / len(o2_errors)),
            "expert_count_copy_baseline_l1": None if not o3_errors else sum(o3_errors) / len(o3_errors),
            "traffic_copy_baseline_l1": None if not o4_errors else sum(o4_errors) / len(o4_errors),
            "best_non_oracle_expert_to_traffic_l1": min(
                [value for value in [mean_relative, None if not o3_errors else sum(o3_errors) / len(o3_errors)] if value is not None],
                default=None,
            ),
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
            "| Layer | O1 source-expert->traffic L1 | O2 global-expert->traffic L1 | O3 current-expert-copy->next L1 | O4 current-traffic-copy->next L1 | cosine | self bytes ignored |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        expert_copy_text = "-" if row["expert_count_copy_baseline_l1"] is None else f"{row['expert_count_copy_baseline_l1']:.4f}"
        traffic_copy_text = "-" if row["traffic_copy_baseline_l1"] is None else f"{row['traffic_copy_baseline_l1']:.4f}"
        lines.append(
            f"| {row['layer_id']} | {row['actual_source_expert_to_traffic_relative_l1']:.4f} | {row['global_expert_count_to_traffic_relative_l1']:.4f} | "
            f"{expert_copy_text} | "
            f"{traffic_copy_text} | "
            f"{row['traffic_cosine']:.4f} | {row['self_bytes_ignored']} |"
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
