#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import socket
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.distributed_runtime import (
    ProtocolConfig,
    build_workload_plan,
    build_placement_map,
    create_workload_plans,
    dependency_score_for_bucket,
    lina_inspired_score_for_bucket,
    state_score_for_bucket,
    strong_state_score_for_bucket,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose whether dependency features add information beyond state-like features.")
    parser.add_argument("--model", choices=["olmoe", "qwen_moe"], default="olmoe")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(ROOT / "artifacts" / "poc2_information_gain_latest.json"))
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--microbatch-count", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    model_path = args.model_path or (
        "/root/autodl-tmp/models/OLMoE-1B-7B-0924" if args.model == "olmoe" else "/root/autodl-tmp/models/Qwen1.5-MoE-A2.7B"
    )
    protocol = ProtocolConfig(
        model_key=args.model,
        model_path=model_path,
        output_dir=str(ROOT / "outputs" / "poc2_information_gain_tmp"),
        artifact_dir=str(ROOT / "artifacts" / "poc2_information_gain_tmp"),
        microbatch_size=args.microbatch_size,
        microbatch_count=args.microbatch_count,
        max_length=args.max_length,
        seed=args.seed,
        world_size=max(1, min(4, __import__("torch").cuda.device_count())),
    )
    plans, manifest = create_workload_plans_local(protocol)
    rows = []
    for plan in plans:
        for bucket in plan.bucket_workloads:
            dep = dependency_score_for_bucket(bucket)
            state = state_score_for_bucket(bucket, None)
            strong = strong_state_score_for_bucket(bucket, None)
            lina = lina_inspired_score_for_bucket(bucket, None)
            outcome = bucket.estimated_service_units * (1.0 + bucket.bridge_score + bucket.coactive_peer_degree)
            rows.append(
                {
                    "plan_id": plan.plan_id,
                    "bucket_id": bucket.bucket_id,
                    "dependency_score": dep,
                    "state_score": state,
                    "strong_state_score": strong,
                    "lina_score": lina,
                    "token_count": bucket.token_count,
                    "payload_bytes": bucket.payload_rows * bucket.hidden_dim * 2,
                    "destination_rank": bucket.destination_rank,
                    "estimated_service_units": bucket.estimated_service_units,
                    "runtime_outcome_proxy": outcome,
                }
            )

    result = {
        "manifest": manifest,
        "row_count": len(rows),
        "correlations": {
            "dependency_vs_token_count": _corr([row["dependency_score"] for row in rows], [row["token_count"] for row in rows]),
            "dependency_vs_payload_bytes": _corr([row["dependency_score"] for row in rows], [row["payload_bytes"] for row in rows]),
            "dependency_vs_estimated_service_units": _corr([row["dependency_score"] for row in rows], [row["estimated_service_units"] for row in rows]),
            "dependency_vs_state_score": _corr([row["dependency_score"] for row in rows], [row["state_score"] for row in rows]),
            "dependency_vs_strong_state_score": _corr([row["dependency_score"] for row in rows], [row["strong_state_score"] for row in rows]),
        },
        "topk_overlap": _topk_overlap(rows),
        "incremental_prediction": _incremental_prediction(rows),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)
    return 0


def create_workload_plans_local(protocol: ProtocolConfig):
    import torch.distributed as dist

    initialized = dist.is_initialized()
    if not initialized:
        import os

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            port = handle.getsockname()[1]
        os.environ.setdefault("MASTER_PORT", str(port))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")
    try:
        return create_workload_plans(protocol)
    finally:
        if not initialized:
            dist.destroy_process_group()


def _corr(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if denom_left == 0.0 or denom_right == 0.0:
        return 0.0
    return numerator / (denom_left * denom_right)


def _topk_overlap(rows: list[dict[str, float]], k: int = 5) -> dict[str, float]:
    dep = {row["bucket_id"] for row in sorted(rows, key=lambda item: item["dependency_score"], reverse=True)[:k]}
    state = {row["bucket_id"] for row in sorted(rows, key=lambda item: item["state_score"], reverse=True)[:k]}
    strong = {row["bucket_id"] for row in sorted(rows, key=lambda item: item["strong_state_score"], reverse=True)[:k]}
    lina = {row["bucket_id"] for row in sorted(rows, key=lambda item: item["lina_score"], reverse=True)[:k]}
    return {
        "dependency_state_jaccard": _jaccard(dep, state),
        "dependency_strong_state_jaccard": _jaccard(dep, strong),
        "dependency_lina_jaccard": _jaccard(dep, lina),
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _incremental_prediction(rows: list[dict[str, float]]) -> dict[str, float]:
    state_errors = []
    state_dep_errors = []
    for row in rows:
        state_pred = 0.7 * row["state_score"] + 0.3 * row["estimated_service_units"]
        state_dep_pred = state_pred + 0.25 * row["dependency_score"]
        outcome = row["runtime_outcome_proxy"]
        state_errors.append(abs(outcome - state_pred))
        state_dep_errors.append(abs(outcome - state_dep_pred))
    state_mae = float(statistics.fmean(state_errors)) if state_errors else 0.0
    state_dep_mae = float(statistics.fmean(state_dep_errors)) if state_dep_errors else 0.0
    return {
        "state_only_mae": state_mae,
        "state_plus_dependency_mae": state_dep_mae,
        "mae_gain": state_mae - state_dep_mae,
    }


if __name__ == "__main__":
    raise SystemExit(main())
