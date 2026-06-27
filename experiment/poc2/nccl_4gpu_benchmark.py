#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.distributed_runtime import (
    benchmark_preflight,
    counterbalanced_orders,
    GlobalRuntimeState,
    ProtocolConfig,
    aggregate_repetition_records,
    broadcast_global_dispatch_plan,
    build_global_dispatch_plan,
    cleanup_nccl,
    create_workload_plans,
    environment_snapshot,
    init_nccl,
    mark_state_meaningful,
    preallocate_execution_cache,
    resolve_strategy_orders,
    run_policy_on_plan,
    save_artifacts,
    save_plan_snapshot,
    update_global_runtime_state,
)


SANITY_STRATEGIES = ["fifo", "random-order", "strong-state", "full"]


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
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--microbatch-count", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--strategy-order-mode", choices=["latin-square"], default="latin-square")
    parser.add_argument("--compute-scale", type=float, default=None)
    parser.add_argument("--hidden-dim-scale", type=float, default=None)
    parser.add_argument("--intermediate-scale", type=float, default=None)
    parser.add_argument("--release-round-size", type=int, default=4)
    parser.add_argument("--regime", choices=["communication-dominant", "balanced", "compute-dominant"], default="balanced")
    parser.add_argument("--audit-plan-id", type=str, default=None)
    parser.add_argument("--plan-snapshot-path", type=str, default=None)
    parser.add_argument("--strategy-subset", nargs="+", default=None)
    parser.add_argument("--origin-sharding", choices=["contiguous", "round-robin"], default="round-robin")
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
            audit_plan_id=args.audit_plan_id,
            audit_strategy_subset=args.strategy_subset,
            plan_snapshot_path=args.plan_snapshot_path,
            origin_sharding=args.origin_sharding,
        )
        plans, workload_manifest = create_workload_plans(protocol)
        active_strategies = list(protocol.audit_strategy_subset or SANITY_STRATEGIES)
        if active_strategies != SANITY_STRATEGIES:
            raise RuntimeError(f"sanity benchmark only allows {SANITY_STRATEGIES}, got {active_strategies}")
        if protocol.repetitions % 4 != 0:
            raise RuntimeError("sanity benchmark requires repetitions to be a multiple of 4")
        warmup_orders = counterbalanced_orders(active_strategies, protocol.warmup_repetitions)
        measured_orders = resolve_strategy_orders(
            protocol.strategy_order_mode,
            protocol.repetitions,
            protocol.seed,
            strategies=active_strategies,
        )
        orders = warmup_orders + measured_orders
        per_repetition: list[dict[str, object]] = []
        global_states = {strategy: GlobalRuntimeState() for strategy in active_strategies}
        execution_caches = {
            plan.plan_id: preallocate_execution_cache(
                plan,
                state["rank"],
                device=__import__("torch").device(f"cuda:{state['local_rank']}"),
                correctness_mode=False,
            )
            for plan in plans
        }
        allowed_pids = [int(os.getpid()), int(os.getppid())]
        gathered_pids: list[object] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_pids, {"pid": int(os.getpid()), "ppid": int(os.getppid())})
        if rank == 0:
            allowed_pids = sorted(
                {
                    int(item["pid"])
                    for item in gathered_pids
                }
                | {
                    int(item["ppid"])
                    for item in gathered_pids
                }
            )
        preflight = benchmark_preflight(
            protocol,
            plans[0],
            execution_caches[plans[0].plan_id],
            allowed_pids=allowed_pids,
            allowed_cmd_substrings=["experiment/poc2/nccl_4gpu_benchmark.py"],
        ) if rank == 0 else None
        fairness_references_by_plan: dict[tuple[int, str], dict[str, object]] = {}

        for repetition_index, order in enumerate(orders):
            warmup = repetition_index < protocol.warmup_repetitions
            for microbatch_id, plan in enumerate(plans):
                fairness_key = (repetition_index, plan.plan_id)
                fairness_reference = fairness_references_by_plan.setdefault(
                    fairness_key,
                    {
                        **dict(preflight or {}),
                        "total_dispatch_matrix": None,
                        "total_return_matrix": None,
                    },
                )
                for strategy in order:
                    local_global_plan = None
                    if rank == 0:
                        local_global_plan = build_global_dispatch_plan(
                            protocol=protocol,
                            plan=plan,
                            strategy=strategy,
                            global_state=global_states[strategy],
                            repetition_index=repetition_index,
                        )
                    global_plan = broadcast_global_dispatch_plan(local_global_plan)
                    result = run_policy_on_plan(
                        protocol=protocol,
                        plan=plan,
                        global_plan=global_plan,
                        execution_cache=execution_caches[plan.plan_id],
                        repetition_index=repetition_index,
                        warmup=warmup,
                        local_rank=state["local_rank"],
                    )
                    result["strategy_order"] = list(order)
                    result["fairness_reference"] = {
                        "plan_hash": fairness_reference.get("plan_hash"),
                        "trace_hash": fairness_reference.get("trace_hash"),
                        "placement_hash": fairness_reference.get("placement_hash"),
                        "token_origin_map_hash": fairness_reference.get("token_origin_map_hash"),
                        "payload_checksum": fairness_reference.get("payload_checksum"),
                        "metadata_checksum": fairness_reference.get("metadata_checksum"),
                        "expert_weight_checksum": fairness_reference.get("expert_weight_checksum"),
                        "total_route_item_set_hash": fairness_reference.get("total_route_item_set_hash"),
                    }
                    if fairness_reference["total_dispatch_matrix"] is None:
                        fairness_reference["total_dispatch_matrix"] = result["dispatch_bytes_matrix"]
                        fairness_reference["total_return_matrix"] = result["return_bytes_matrix"]
                    else:
                        if result["dispatch_bytes_matrix"] != fairness_reference["total_dispatch_matrix"]:
                            raise RuntimeError("fairness assertion failed: total dispatch matrix changed across strategies")
                        if result["return_bytes_matrix"] != fairness_reference["total_return_matrix"]:
                            raise RuntimeError("fairness assertion failed: total return matrix changed across strategies")
                    per_repetition.append(result)
                    rank_payload = {
                        "rank": state["rank"],
                        "local_completion_ms": float(result["local_end_to_end_completion_ms"]),
                        "dispatch_total_ms": float(result["dispatch_total_ms"]),
                        "compute_total_ms": float(result["compute_total_ms"]),
                        "return_total_ms": float(result["return_total_ms"]),
                    }
                    rank_records: list[object] = [None for _ in range(world_size)]
                    dist.all_gather_object(rank_records, rank_payload)
                    if rank == 0:
                        global_states[strategy] = update_global_runtime_state(
                            global_states[strategy],
                            global_plan,
                            plan,
                            rank_records,  # type: ignore[arg-type]
                        )
                    state_payload = [global_states[strategy].to_dict() if rank == 0 else None]
                    dist.broadcast_object_list(state_payload, src=0)
                    global_states[strategy] = GlobalRuntimeState.from_dict(state_payload[0])

        summary = aggregate_repetition_records(per_repetition, mark_state_meaningful(protocol.microbatch_count))
        if rank == 0:
            plan_snapshot = save_plan_snapshot(protocol.artifact_dir, plans)
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
                    "origin_sharding": protocol.origin_sharding,
                    "audit_plan_id": protocol.audit_plan_id,
                    "plan_snapshot_path": str(plan_snapshot),
                    "strategy_subset": active_strategies,
                    "sanity_benchmark_only": True,
                    "paper_claim_ready": False,
                },
                environment=environment_snapshot(protocol.world_size),
                protocol={
                    "backend": "nccl-dispatch",
                    "real_routing": True,
                    "dispatch_backend": "torch.distributed.all_to_all_single",
                    "return_backend": "torch.distributed.all_to_all_single",
                    "expert_compute": "rank-local expert-like MLP",
                    "origin_sharding": protocol.origin_sharding,
                    "control_plane": "rank0 only",
                    "data_plane": "all ranks execute received global dispatch plan only",
                    "strategy_order_mode": protocol.strategy_order_mode,
                    "latin_square_orders": orders,
                    "benchmark_entrypoint": "experiment/poc2/nccl_4gpu_benchmark.py",
                    "policies": active_strategies,
                    "preflight": preflight,
                    "fairness_references_by_plan": {
                        f"rep{repetition_index}:{plan_id}": value
                        for (repetition_index, plan_id), value in fairness_references_by_plan.items()
                    },
                    "sanity_benchmark_only": True,
                    "paper_claim_ready": False,
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
