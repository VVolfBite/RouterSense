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
    init_nccl,
    preallocate_execution_cache,
    run_policy_on_plan,
)


def _plan() -> WorkloadPlan:
    buckets = [
        WorkloadBucket(
            bucket_id="b0",
            route_id="r0",
            token_ids=[0],
            token_positions=[0],
            origin_rank=0,
            destination_id=10,
            destination_rank=1,
            expert_id=10,
            layer_id=0,
            microbatch_id=0,
            token_count=1,
            payload_rows=1,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=1.0,
            payload_bytes=128,
            source_count=1,
            source_coverage=0.1,
            coactive_peer_degree=0.1,
            coactive_event_density=0.1,
            position_spread=0.0,
            bridge_score=0.1,
            route_share=0.1,
            density_over_mean=0.1,
            size_norm=0.1,
            inverse_size_rank_norm=1.0,
            is_hot_bucket=False,
        ),
        WorkloadBucket(
            bucket_id="b1",
            route_id="r1",
            token_ids=[1],
            token_positions=[1],
            origin_rank=2,
            destination_id=11,
            destination_rank=1,
            expert_id=11,
            layer_id=0,
            microbatch_id=0,
            token_count=1,
            payload_rows=1,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=1.0,
            payload_bytes=128,
            source_count=1,
            source_coverage=0.1,
            coactive_peer_degree=0.1,
            coactive_event_density=0.1,
            position_spread=0.0,
            bridge_score=0.1,
            route_share=0.1,
            density_over_mean=0.1,
            size_norm=0.1,
            inverse_size_rank_norm=0.5,
            is_hot_bucket=False,
        ),
    ]
    return WorkloadPlan(
        plan_id="expert_grouping_smoke",
        model_key="synthetic_verify",
        seed=1,
        microbatch_id=0,
        layer_id=0,
        layer_path="layer",
        hidden_dim=64,
        intermediate_dim=128,
        placement_policy="fixed",
        origin_sharding="round-robin",
        placement_mapping=[PlacementEntry(expert_id=10, destination_rank=1), PlacementEntry(expert_id=11, destination_rank=1)],
        bucket_workloads=buckets,
        routing_summary={},
    )


def main() -> int:
    state = init_nccl()
    rank = state["rank"]
    try:
        plan = _plan()
        protocol = ProtocolConfig(
            model_key="synthetic_verify",
            model_path="",
            output_dir=str(ROOT / "outputs" / "poc2_expert_grouping_smoke"),
            artifact_dir=str(ROOT / "artifacts" / "poc2_expert_grouping_smoke"),
            world_size=state["world_size"],
            release_round_size=2,
            require_remote_traffic=False,
        )
        local_plan = build_global_dispatch_plan(
            protocol=protocol,
            plan=plan,
            strategy="fifo",
            global_state=GlobalRuntimeState(),
            repetition_index=0,
        ) if rank == 0 else None
        global_plan = broadcast_global_dispatch_plan(local_plan)
        cache = preallocate_execution_cache(plan, rank, torch.device(f"cuda:{state['local_rank']}"), correctness_mode=True)
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
            expert_records = []
            for round_item in result["rounds"]:
                expert_records.extend(round_item["expert_group_records"])
            print(json.dumps({"expert_group_records": expert_records}, indent=2))
        return 0
    finally:
        cleanup_nccl()


if __name__ == "__main__":
    raise SystemExit(main())
