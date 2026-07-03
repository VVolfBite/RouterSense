#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

from _bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

import torch  # type: ignore

from rs.runtime import load_model_and_tokenizer
from rs.runtime.distributed_ep.adapter.runner import (
    DistributedRunnerConfig,
    build_distributed_runner_plan,
    execute_scheduled_inference,
)
from rs.trace import collect_full_sequence_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run native baseline vs wave-collective OLMoE execution.")
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="U_gated_maxweight_matching")
    parser.add_argument("--prompt", type=str, default="Explain mixture-of-experts routing in one paragraph.")
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--execution-mode", choices=["native_baseline", "wave_collective"], default="native_baseline")
    parser.add_argument("--compute-mode", choices=["actual_olmoe_expert", "simulated_delay"], default="actual_olmoe_expert")
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    parser.add_argument("--layer-index", type=int, default=0, help="Index into MoE layer ids, not raw transformer layer id.")
    parser.add_argument("--max-waves", type=int, default=0, help="Cap the number of execution waves; 0 means uncapped.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "wave_execution"))
    args = parser.parse_args(argv)

    import torch.distributed as dist  # type: ignore

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    model, tokenizer, _, _, _ = load_model_and_tokenizer(
        model_id=args.model,
        model_path=args.model_path,
        precision=args.precision,
        device_index=local_rank,
    )
    trace = collect_full_sequence_trace(
        model,
        tokenizer,
        args.prompt,
        request_id="wave-exec",
        sample_id="wave-sample",
    )
    moe_layer_ids = list(trace["summary"].get("moe_layer_ids", []))
    if not moe_layer_ids:
        raise RuntimeError("trace returned no MoE layer ids")
    if args.layer_index < 0 or args.layer_index >= len(moe_layer_ids):
        raise RuntimeError(f"layer_index {args.layer_index} out of range for {len(moe_layer_ids)} MoE layers")
    layer_id = int(moe_layer_ids[args.layer_index])

    runner_plan = build_distributed_runner_plan(
        model=model,
        trace=trace,
        config=DistributedRunnerConfig(
            world_size=world_size,
            node_rank=int(os.environ.get("NODE_RANK", 0)),
            model_id=args.model,
            origin_rank=0,
        ),
        rank=rank,
        host=socket.gethostname(),
        gpu_name=f"cuda:{local_rank}",
    )
    hidden_state_rows = trace["hidden_states"][layer_id][0].to(dtype=torch.float16, device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    plan_index = runner_plan.dispatch_plans.index(next(plan for plan in runner_plan.dispatch_plans if int(plan.layer_id) == layer_id))

    started = time.perf_counter()
    execution = execute_scheduled_inference(
        dispatch_plans=runner_plan.dispatch_plans,
        rank=rank,
        world_size=world_size,
        strategy_name=args.strategy,
        hidden_size=int(runner_plan.adapter.get("hidden_size", hidden_state_rows.shape[-1])),
        expert_compute_delay=args.expert_compute_delay if args.compute_mode == "simulated_delay" else 0.0,
        use_distributed=True,
        execution_mode=args.execution_mode,
        local_expert_weights=runner_plan.local_expert_weight_bundle,
        hidden_state_rows=hidden_state_rows,
        plan_index=plan_index,
        max_waves=(args.max_waves if args.max_waves > 0 else None),
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    payload = {
        "rank": rank,
        "world_size": world_size,
        "model": args.model,
        "execution_mode": args.execution_mode,
        "compute_mode": args.compute_mode,
        "strategy": args.strategy,
        "layer_id": layer_id,
        "prompt": args.prompt,
        "wall_ms": wall_ms,
        "execution": execution,
    }
    gathered: list[dict] | None = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(payload, gathered, dst=0)
    if rank == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = {
            "run": {
                "model": args.model,
                "strategy": args.strategy,
                "execution_mode": args.execution_mode,
                "compute_mode": args.compute_mode,
                "layer_id": layer_id,
                "world_size": world_size,
                "max_waves": args.max_waves,
            },
            "ranks": gathered,
        }
        (out / f"{args.execution_mode}_{args.strategy}_layer{layer_id}.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
