#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

from _bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime import load_model_and_tokenizer, run_single_gpu_text_inference
from rs.runtime.distributed_ep.adapter.runner import (
    DistributedRunnerConfig,
    build_distributed_runner_plan,
    simulate_rank_execution,
)
from rs.trace import collect_olmoe_router_trace, summarize_router_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0C OLMoE EP planning smoke on one visible rank.")
    parser.add_argument("--model-id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--inventory", type=str, default=str(ROOT / "deploy" / "inventory" / "hosts.local.yaml"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="The history of science is a story of")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--gpu-name", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "phase0c_distributed_ep_smoke"))
    args = parser.parse_args(argv)
    model_path = args.model_path
    if model_path is None:
        model_path = os.environ.get("RS_MODEL_PATH")
    model, tokenizer, _, _, _ = load_model_and_tokenizer(
        model_id=args.model_id,
        model_path=model_path,
        precision=args.precision,
        device_index=args.device_index,
    )
    trace = collect_olmoe_router_trace(model, tokenizer, args.prompt)
    summary = summarize_router_trace(trace)
    reference = run_single_gpu_text_inference(
        model_id=args.model_id,
        model_path=model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
        device_index=args.device_index,
    )
    runner_config = DistributedRunnerConfig(
        world_size=args.world_size,
        node_rank=args.node_rank,
        model_id=args.model_id,
        origin_rank=args.rank,
    )
    runner_plan = build_distributed_runner_plan(
        model=model,
        trace=trace,
        config=runner_config,
        rank=args.rank,
        host=socket.gethostname(),
        gpu_name=args.gpu_name,
    )
    simulated = simulate_rank_execution(
        runner_plan.dispatch_plans,
        rank=args.rank,
        bytes_per_row=int(runner_plan.adapter["hidden_size"]) * 2,
    )
    payload = {
        "inventory": args.inventory,
        "trace_summary": summary,
        "reference": reference.to_dict(),
        "distributed_runner_config": runner_config.to_dict(),
        "runner_plan": {
            "adapter": runner_plan.adapter,
            "placement": runner_plan.placement,
            "residency": runner_plan.residency,
            "dispatch_summary": runner_plan.dispatch_summary,
            "dispatch_plans": [plan.to_dict() for plan in runner_plan.dispatch_plans],
            "local_expert_weights": runner_plan.local_expert_weights,
            "manifest": runner_plan.manifest,
        },
        "simulated_rank0_execution": simulated,
        "status": "PHASE0C_PRE_GPU_EXECUTION_PLAN_READY",
        "note": "This entrypoint validates planning and local execution wiring on one visible rank. Use exp_wave_execution.py or distributed_nccl_smoke.py for real multi-rank execution.",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "distributed_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
