#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.distributed_runtime import (
    GlobalRuntimeState,
    ProtocolConfig,
    broadcast_global_dispatch_plan,
    build_global_dispatch_plan,
    cleanup_nccl,
    create_workload_plans,
    environment_snapshot,
    init_nccl,
    preallocate_execution_cache,
    run_policy_on_plan,
    summarize_route_manifests,
)


MODEL_PATHS = {
    "olmoe": "/root/autodl-tmp/models/OLMoE-1B-7B-0924",
    "qwen_moe": "/root/autodl-tmp/models/Qwen1.5-MoE-A2.7B",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate remote dispatch semantics using a real router trace.")
    parser.add_argument("--model", choices=sorted(MODEL_PATHS), default="olmoe")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--placement-policy", choices=["modulo", "balanced", "hotspot"], default="balanced")
    parser.add_argument("--origin-sharding", choices=["contiguous", "round-robin"], default="round-robin")
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--microbatch-count", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--artifact-dir", type=str, default=str(ROOT / "artifacts" / "poc2_real_trace_semantic_smoke"))
    args = parser.parse_args(argv)

    state = init_nccl()
    rank = state["rank"]
    try:
        protocol = ProtocolConfig(
            model_key=args.model,
            model_path=args.model_path or MODEL_PATHS[args.model],
            output_dir=str(ROOT / "outputs" / "poc2_real_trace_semantic_smoke"),
            artifact_dir=args.artifact_dir,
            placement_policy=args.placement_policy,
            world_size=state["world_size"],
            microbatch_size=args.microbatch_size,
            microbatch_count=args.microbatch_count,
            max_length=args.max_length,
            warmup_repetitions=0,
            repetitions=1,
            release_round_size=999999,
            origin_sharding=args.origin_sharding,
            require_remote_traffic=True,
        )
        plans, manifest = create_workload_plans(protocol)
        if not plans:
            raise RuntimeError("no workload plans created from real router trace")
        global_state = GlobalRuntimeState()
        local_plan = build_global_dispatch_plan(
            protocol=protocol,
            plan=plans[0],
            strategy="fifo",
            global_state=global_state,
            repetition_index=0,
        ) if rank == 0 else None
        global_plan = broadcast_global_dispatch_plan(local_plan)
        cache = preallocate_execution_cache(
            plans[0],
            rank,
            torch.device(f"cuda:{state['local_rank']}"),
            correctness_mode=True,
        )
        result = run_policy_on_plan(
            protocol=protocol,
            plan=plans[0],
            global_plan=global_plan,
            execution_cache=cache,
            repetition_index=0,
            warmup=False,
            local_rank=state["local_rank"],
        )
        if rank == 0:
            artifact_dir = Path(protocol.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            manifest_summary = summarize_route_manifests(
                result["dispatch_route_manifest"],
                result["receive_route_manifest"],
                result["return_route_manifest"],
                result["origin_verified_manifest"],
            )
            payload = {
                "kind": "remote_dispatch_semantic_validation",
                "not_a_benchmark": True,
                "model": args.model,
                "placement_policy": args.placement_policy,
                "origin_sharding": args.origin_sharding,
                "dispatch_bytes_matrix": result["dispatch_bytes_matrix"],
                "return_bytes_matrix": result["return_bytes_matrix"],
                "remote_byte_ratio": result["remote_byte_ratio"],
                "communication_semantics_valid": result["communication_semantics_valid"],
                "decision_hash": result["decision_hash"],
                "used_python_metadata_gather": result["used_python_metadata_gather"],
                **manifest_summary,
                "workload_manifest": manifest,
                "environment": environment_snapshot(protocol.world_size),
            }
            (artifact_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            for name, items in [
                ("dispatch_route_manifest.jsonl", result["dispatch_route_manifest"]),
                ("receive_route_manifest.jsonl", result["receive_route_manifest"]),
                ("return_route_manifest.jsonl", result["return_route_manifest"]),
                ("origin_verified_manifest.jsonl", result["origin_verified_manifest"]),
            ]:
                with (artifact_dir / name).open("w", encoding="utf-8") as handle:
                    for item in items:
                        handle.write(json.dumps(item) + "\n")
            (artifact_dir / "goal.md").write_text(
                "# POC2.5 Real Trace Semantic Smoke\n\n"
                "This run only validates that a real router trace produces non-diagonal origin->destination traffic under the repaired runtime.\n"
                "It is not a performance benchmark and must not be used for policy conclusions.\n",
                encoding="utf-8",
            )
            print(json.dumps(payload, indent=2))
        return 0
    finally:
        cleanup_nccl()


if __name__ == "__main__":
    raise SystemExit(main())
