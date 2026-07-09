#!/usr/bin/env python3
"""Prepare a dry-run checklist for GPU expert-trace collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture-expert-trace", action="store_true", default=False)
    parser.add_argument("--capture-per-token-trace", action="store_true", default=False)
    parser.add_argument("--output-dir", default="outputs/comparison/tmp_prediction_probe")
    return parser.parse_args()


def build_checklist(*, capture_expert_trace: bool, capture_per_token_trace: bool, output_dir: str) -> dict[str, object]:
    return {
        "capture_expert_trace": bool(capture_expert_trace),
        "capture_per_token_trace": bool(capture_per_token_trace),
        "heavy_debug_trace": bool(capture_per_token_trace),
        "fast_path_clean": not bool(capture_per_token_trace),
        "output_dir": str(output_dir),
        "required_output_files": [
            "rank*_expert_route_trace.jsonl",
            "rank*_source_expert_counts.jsonl",
            "rank*_expert_to_traffic_audit.jsonl",
            "rank*_expert_trace_warnings.jsonl",
            "rank*_transport_execution.jsonl",
            "rank*_prediction_audit.jsonl",
        ],
        "required_fields": [
            "layer_id",
            "rank",
            "source_rank",
            "selected_experts_available",
            "routing_weights_available",
            "expert_to_rank_map",
            "source_expert_counts",
            "bytes_per_token",
        ],
        "success_criteria": {
            "non_empty_source_expert_counts": True,
            "layer_count_min": 2,
            "rank_count_min": 4,
            "expert_to_traffic_reconstruction_runnable": True,
        },
        "next_gpu_steps": [
            "collect expert route trace",
            "verify source_expert_counts non-empty",
            "run expert_to_traffic reconstruction",
            "compare O1/O2/O3/O4 baselines",
            "only then decide whether to implement real gate replay predictor",
        ],
    }


def main() -> None:
    args = _parse_args()
    payload = build_checklist(
        capture_expert_trace=bool(args.capture_expert_trace),
        capture_per_token_trace=bool(args.capture_per_token_trace),
        output_dir=str(args.output_dir),
    )
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
