#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # type: ignore

from routesense.runtime import load_model_and_tokenizer
from routesense.runtime.distributed_ep.adapter.runner import (
    DistributedRunnerConfig,
    _build_matrix_from_plan,
    build_distributed_runner_plan,
    execute_scheduled_inference,
    simulate_rank_execution,
)
from routesense.scheduler import SchedulingContext, get_strategy
from routesense.trace import collect_olmoe_router_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run scheduled distributed EP execution experiments.")
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="best_of")
    parser.add_argument("--strategies", type=str, default="")
    parser.add_argument("--prompt", type=str, default="Hello, world!")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    parser.add_argument("--single-gpu", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "scheduling"))
    args = parser.parse_args(argv)

    strategies = [item for item in args.strategies.split(",") if item] if args.strategies else [args.strategy]
    if args.single_gpu:
        return _run_single_gpu(args, strategies)
    return _run_torchrun(args, strategies)


def _run_torchrun(args, strategies: list[str]) -> int:
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
    trace = collect_olmoe_router_trace(model, tokenizer, args.prompt)
    config = DistributedRunnerConfig(
        world_size=world_size,
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        model_id=args.model,
        origin_rank=0,
    )
    runner_plan = build_distributed_runner_plan(
        model=model,
        trace=trace,
        config=config,
        rank=rank,
        host=socket.gethostname(),
        gpu_name=f"cuda:{local_rank}",
    )

    per_rank_results: dict[str, dict] = {}
    for strategy_name in strategies:
        dist.barrier()
        t0 = time.perf_counter()
        execution = execute_scheduled_inference(
            dispatch_plans=runner_plan.dispatch_plans,
            rank=rank,
            world_size=world_size,
            strategy_name=strategy_name,
            hidden_size=int(runner_plan.adapter.get("hidden_size", 2048)),
            expert_compute_delay=args.expert_compute_delay,
        )
        per_rank_results[strategy_name] = {
            **execution,
            "wall_time_s": time.perf_counter() - t0,
        }

    rank_payload = {
        "rank": rank,
        "host": socket.gethostname(),
        "gpu_name": f"cuda:{local_rank}",
        "results": per_rank_results,
        "dispatch_summary": runner_plan.dispatch_summary,
    }
    gathered: list[dict] | None = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(rank_payload, gathered, dst=0)

    if rank == 0:
        payload = {
            "world_size": world_size,
            "model": args.model,
            "strategies": strategies,
            "adapter": runner_plan.adapter,
            "placement": runner_plan.placement,
            "dispatch_summary": runner_plan.dispatch_summary,
            "ranks": gathered,
        }
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scheduled_execution.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))

    dist.destroy_process_group()
    return 0


def _run_single_gpu(args, strategies: list[str]) -> int:
    model, tokenizer, _, _, _ = load_model_and_tokenizer(
        model_id=args.model,
        model_path=args.model_path,
        precision=args.precision,
        device_index=0,
    )
    trace = collect_olmoe_router_trace(model, tokenizer, args.prompt)
    config = DistributedRunnerConfig(world_size=4, node_rank=0, model_id=args.model, origin_rank=0)
    runner_plan = build_distributed_runner_plan(
        model=model,
        trace=trace,
        config=config,
        rank=0,
        host=socket.gethostname(),
        gpu_name="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    dispatch_matrix = _build_matrix_from_plan(runner_plan.dispatch_plans[0], 0, "send")
    combine_matrix = _build_matrix_from_plan(runner_plan.dispatch_plans[0], 0, "recv")
    next_matrix = (
        _build_matrix_from_plan(runner_plan.dispatch_plans[1], 0, "send")
        if len(runner_plan.dispatch_plans) > 1
        else [[0] * config.world_size for _ in range(config.world_size)]
    )

    results: dict[str, dict] = {}
    for strategy_name in strategies:
        strategy = get_strategy(strategy_name)
        ctx = SchedulingContext(
            dispatch_matrix=dispatch_matrix,
            combine_matrix=combine_matrix,
            next_dispatch_matrix=next_matrix,
            num_gpus=config.world_size,
            model="full_duplex",
            expert_compute_delay=args.expert_compute_delay,
        )
        scheduled = strategy.solve(ctx)
        results[strategy_name] = {
            "makespan": scheduled.makespan,
            "solve_time_ms": scheduled.solve_time_ms,
            "chunk_count": len(scheduled.schedule),
        }

    payload = {
        "mode": "single_gpu_debug",
        "model": args.model,
        "strategies": strategies,
        "dispatch_matrix": dispatch_matrix,
        "combine_matrix": combine_matrix,
        "next_dispatch_matrix": next_matrix,
        "results": results,
        "simulated_execution": simulate_rank_execution(
            runner_plan.dispatch_plans,
            rank=0,
            bytes_per_row=int(runner_plan.adapter.get("hidden_size", 2048)) * 2,
        ),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scheduled_execution_debug.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
