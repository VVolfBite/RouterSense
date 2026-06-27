#!/usr/bin/env python3
from __future__ import annotations

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
    PlacementEntry,
    ProtocolConfig,
    WorkloadBucket,
    WorkloadPlan,
    broadcast_global_dispatch_plan,
    build_global_dispatch_plan,
    cleanup_nccl,
    environment_snapshot,
    init_nccl,
    preallocate_execution_cache,
    run_policy_on_plan,
)


def _fixed_plan(world_size: int) -> WorkloadPlan:
    if world_size != 4:
        raise RuntimeError("verify_remote_dispatch_4gpu.py requires world_size=4")
    flows = [
        (0, 1, 101, [0]),
        (0, 2, 102, [1]),
        (1, 0, 103, [2]),
        (1, 3, 104, [3]),
        (2, 0, 105, [4]),
        (2, 1, 106, [5]),
        (3, 1, 107, [6]),
        (3, 2, 108, [7]),
    ]
    buckets: list[WorkloadBucket] = []
    for index, (origin, destination, expert_id, token_ids) in enumerate(flows):
        rows = 2 + (index % 3)
        buckets.append(
            WorkloadBucket(
                bucket_id=f"fixed_b{index}",
                route_id=f"fixed_r{index}",
                token_ids=list(token_ids),
                token_positions=list(token_ids),
                origin_rank=origin,
                destination_id=expert_id,
                destination_rank=destination,
                expert_id=expert_id,
                layer_id=0,
                microbatch_id=0,
                token_count=len(token_ids),
                payload_rows=rows,
                hidden_dim=64,
                intermediate_dim=128,
                estimated_service_units=float(rows),
                payload_bytes=rows * 64 * 2,
                source_count=8,
                source_coverage=0.25,
                coactive_peer_degree=0.25,
                coactive_event_density=0.25,
                position_spread=0.0,
                bridge_score=0.25,
                route_share=0.125,
                density_over_mean=0.5,
                size_norm=0.5,
                inverse_size_rank_norm=1.0 - index / 7.0,
                is_hot_bucket=False,
            )
        )
    return WorkloadPlan(
        plan_id="fixed_remote_dispatch_4gpu",
        model_key="synthetic_verify",
        seed=7,
        microbatch_id=0,
        layer_id=0,
        layer_path="verify.layer",
        hidden_dim=64,
        intermediate_dim=128,
        placement_policy="fixed",
        origin_sharding="fixed",
        placement_mapping=[PlacementEntry(expert_id=100 + i, destination_rank=i % 4) for i in range(8)],
        bucket_workloads=buckets,
        routing_summary={"kind": "fixed_remote_dispatch_4gpu"},
    )


def main() -> int:
    state = init_nccl()
    rank = state["rank"]
    try:
        plan = _fixed_plan(state["world_size"])
        protocol = ProtocolConfig(
            model_key="synthetic_verify",
            model_path="",
            output_dir=str(ROOT / "outputs" / "poc2_verify_remote_dispatch_4gpu"),
            artifact_dir=str(ROOT / "artifacts" / "poc2_verify_remote_dispatch_4gpu"),
            world_size=state["world_size"],
            release_round_size=8,
            origin_sharding="fixed",
            require_remote_traffic=True,
        )
        global_state = GlobalRuntimeState()
        local_plan = build_global_dispatch_plan(
            protocol=protocol,
            plan=plan,
            strategy="fifo",
            global_state=global_state,
            repetition_index=0,
        ) if rank == 0 else None
        global_plan = broadcast_global_dispatch_plan(local_plan)
        cache = preallocate_execution_cache(
            plan,
            rank,
            torch.device(f"cuda:{state['local_rank']}"),
            correctness_mode=True,
        )
        result = run_policy_on_plan(
            protocol=protocol,
            plan=plan,
            global_plan=global_plan,
            execution_cache=cache,
            repetition_index=0,
            warmup=False,
            local_rank=state["local_rank"],
        )
        if rank == 0:
            artifact_dir = Path(protocol.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            expected_route_item_count = sum(len(bucket.route_item_ids) if bucket.route_item_ids else 1 for bucket in plan.bucket_workloads)
            summary = {
                "communication_semantics_valid": result["communication_semantics_valid"],
                "dispatch_bytes_matrix": result["dispatch_bytes_matrix"],
                "return_bytes_matrix": result["return_bytes_matrix"],
                "remote_byte_ratio": result["remote_byte_ratio"],
                "decision_hash": result["decision_hash"],
                "used_python_metadata_gather": result["used_python_metadata_gather"],
                "expected_route_item_count": expected_route_item_count,
                "dispatch_route_item_count": len(result["dispatch_route_manifest"]),
                "receive_route_item_count": len(result["receive_route_manifest"]),
                "return_route_item_count": len(result["return_route_manifest"]),
                "verified_route_item_count": len(result["origin_verified_manifest"]),
                "environment": environment_snapshot(protocol.world_size),
            }
            (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
                "# POC2.5 Remote Dispatch Verification\n\n"
                "This run validates origin->destination->origin remote dispatch semantics on 4 GPUs.\n"
                "It is not a benchmark and does not support any scheduler performance conclusion.\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2))
        return 0
    finally:
        cleanup_nccl()


if __name__ == "__main__":
    raise SystemExit(main())
