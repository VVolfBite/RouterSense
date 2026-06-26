#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.distributed_runtime import (
    ProtocolConfig,
    STRATEGIES,
    aggregate_repetition_records,
    cleanup_nccl,
    create_workload_plans,
    environment_snapshot,
    init_nccl,
    mark_state_meaningful,
    resolve_strategy_orders,
    run_policy_on_plan,
    save_artifacts,
)
from routesense_poc2.scheduler import RuntimeState


MODEL_PATHS = {
    "olmoe": "/root/autodl-tmp/models/OLMoE-1B-7B-0924",
    "qwen_moe": "/root/autodl-tmp/models/Qwen1.5-MoE-A2.7B",
}


def _regime_defaults(regime: str) -> dict[str, float]:
    if regime == "communication-dominant":
        return {
            "compute_scale": 1.25,
            "hidden_dim_scale": 0.25,
            "intermediate_scale": 1.0,
        }
    if regime == "compute-dominant":
        return {
            "compute_scale": 1.0,
            "hidden_dim_scale": 0.75,
            "intermediate_scale": 3.0,
        }
    return {
        "compute_scale": 1.0,
        "hidden_dim_scale": 0.5,
        "intermediate_scale": 2.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark real routing + NCCL dispatch/return MoE execution harness.")
    parser.add_argument("--model", choices=sorted(MODEL_PATHS), default="olmoe")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "poc2_nccl_latest"))
    parser.add_argument("--artifact-dir", type=str, default=str(ROOT / "artifacts" / "poc2_nccl_latest"))
    parser.add_argument("--prompt-path", type=str, default=None)
    parser.add_argument("--placement-policy", choices=["modulo", "balanced", "hotspot"], default="balanced")
    parser.add_argument("--scenario", choices=["natural", "conflict"], default="natural")
    parser.add_argument("--layer", type=str, default="auto")
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--microbatch-count", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-repetitions", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--strategy-order-mode", choices=["fixed", "randomized", "counterbalanced"], default="counterbalanced")
    parser.add_argument("--compute-scale", type=float, default=None)
    parser.add_argument("--hidden-dim-scale", type=float, default=None)
    parser.add_argument("--intermediate-scale", type=float, default=None)
    parser.add_argument("--release-round-size", type=int, default=4)
    parser.add_argument("--regime", choices=["communication-dominant", "balanced", "compute-dominant"], default="balanced")
    args = parser.parse_args(argv)
    regime_defaults = _regime_defaults(args.regime)

    state = init_nccl()
    rank = state["rank"]
    world_size = state["world_size"]
    try:
        protocol = ProtocolConfig(
            model_key=args.model,
            model_path=args.model_path or MODEL_PATHS[args.model],
            output_dir=args.output_dir,
            artifact_dir=args.artifact_dir,
            placement_policy=args.placement_policy,
            scenario=args.scenario,
            regime=args.regime,
            prompt_path=args.prompt_path,
            layer=args.layer,
            microbatch_size=args.microbatch_size,
            microbatch_count=args.microbatch_count,
            max_length=args.max_length,
            precision=args.precision,
            top_k=args.top_k,
            seed=args.seed,
            world_size=world_size,
            warmup_repetitions=args.warmup_repetitions,
            repetitions=args.repetitions,
            strategy_order_mode=args.strategy_order_mode,
            compute_scale=args.compute_scale if args.compute_scale is not None else regime_defaults["compute_scale"],
            hidden_dim_scale=args.hidden_dim_scale if args.hidden_dim_scale is not None else regime_defaults["hidden_dim_scale"],
            intermediate_scale=args.intermediate_scale if args.intermediate_scale is not None else regime_defaults["intermediate_scale"],
            release_round_size=args.release_round_size,
        )
        plans, workload_manifest = create_workload_plans(protocol)
        orders = resolve_strategy_orders(protocol.strategy_order_mode, protocol.warmup_repetitions + protocol.repetitions, protocol.seed)
        per_repetition: list[dict[str, object]] = []

        for repetition_index, order in enumerate(orders):
            warmup = repetition_index < protocol.warmup_repetitions
            runtime_states = {strategy: RuntimeState() for strategy in STRATEGIES}
            for microbatch_id, plan in enumerate(plans):
                for strategy in order:
                    result = run_policy_on_plan(
                        protocol=protocol,
                        plan=plan,
                        strategy=strategy,
                        runtime_state=runtime_states[strategy],
                        repetition_index=repetition_index,
                        warmup=warmup,
                        local_rank=state["local_rank"],
                    )
                    result["strategy_order"] = list(order)
                    per_repetition.append(result)

        summary = aggregate_repetition_records(per_repetition, mark_state_meaningful(protocol.microbatch_count))
        if rank == 0:
            save_artifacts(
                protocol.artifact_dir,
                summary=summary,
                config={
                    "model": protocol.model_key,
                    "model_path": protocol.model_path,
                    "placement_policy": protocol.placement_policy,
                    "scenario": protocol.scenario,
                    "regime": protocol.regime,
                    "world_size": protocol.world_size,
                    "warmup_repetitions": protocol.warmup_repetitions,
                    "repetitions": protocol.repetitions,
                    "strategy_order_mode": protocol.strategy_order_mode,
                    "microbatch_size": protocol.microbatch_size,
                    "microbatch_count": protocol.microbatch_count,
                    "max_length": protocol.max_length,
                    "seed": protocol.seed,
                    "compute_scale": protocol.compute_scale,
                    "hidden_dim_scale": protocol.hidden_dim_scale,
                    "intermediate_scale": protocol.intermediate_scale,
                    "release_round_size": protocol.release_round_size,
                },
                environment=environment_snapshot(protocol.world_size),
                protocol={
                    "backend": "nccl-dispatch",
                    "real_routing": True,
                    "dispatch_backend": "torch.distributed.all_to_all_single",
                    "return_backend": "torch.distributed.all_to_all_single",
                    "expert_compute": "rank-local expert-like MLP",
                    "strategy_order_mode": protocol.strategy_order_mode,
                    "benchmark_entrypoint": "experiment/poc2/nccl_4gpu_benchmark.py",
                    "policies": list(STRATEGIES),
                },
                workload_manifest=workload_manifest,
                per_repetition=per_repetition,
            )
            Path(protocol.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(protocol.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(Path(protocol.artifact_dir))
        return 0
    except Exception as exc:  # pragma: no cover - exercised in real runs
        message = {
            "rank": rank,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if rank == 0:
            print(json.dumps(message, indent=2), file=sys.stderr)
        raise
    finally:
        cleanup_nccl()


if __name__ == "__main__":
    raise SystemExit(main())
