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
    ProtocolConfig,
    broadcast_global_dispatch_plan,
    build_global_dispatch_plan,
    cleanup_nccl,
    create_workload_plans,
    init_nccl,
    preallocate_execution_cache,
    run_policy_on_plan,
)


MODEL_PATHS = {
    "olmoe": "/root/autodl-tmp/models/OLMoE-1B-7B-0924",
}


def main() -> int:
    state = init_nccl()
    rank = state["rank"]
    try:
        protocol = ProtocolConfig(
            model_key="olmoe",
            model_path=MODEL_PATHS["olmoe"],
            output_dir=str(ROOT / "outputs" / "poc2_global_plan_consistency"),
            artifact_dir=str(ROOT / "artifacts" / "poc2_global_plan_consistency"),
            world_size=state["world_size"],
            microbatch_size=2,
            microbatch_count=1,
            max_length=64,
            release_round_size=4,
            origin_sharding="round-robin",
            require_remote_traffic=True,
        )
        plans, _ = create_workload_plans(protocol)
        plan = plans[0]
        global_state = GlobalRuntimeState()
        local_plan = build_global_dispatch_plan(
            protocol=protocol,
            plan=plan,
            strategy="fifo",
            global_state=global_state,
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
            payload = {
                "plan_id": global_plan.plan_id,
                "decision_hash": global_plan.decision_hash,
                "round_count": len(global_plan.release_rounds),
                "communication_semantics_valid": result["communication_semantics_valid"],
            }
            print(json.dumps(payload, indent=2))
        return 0
    finally:
        cleanup_nccl()


if __name__ == "__main__":
    raise SystemExit(main())
