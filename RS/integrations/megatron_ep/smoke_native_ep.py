#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.native_runtime import (
    attach_dispatch_facade,
    attach_dispatch_observer,
    build_position_ids,
    destroy_distributed,
    dtype_from_name,
    gather_rank_payloads,
    init_distributed,
    load_prompts,
    stage_barrier,
    summarize_native_dispatchers,
    summarize_observer_rows,
    summarize_rank_environment,
    validate_observer_mode,
)
from integrations.megatron_ep.routersense.contracts import NativeEPSummary, RouterSenseInjectionConfig
from integrations.megatron_ep.routersense.observer import RouterSenseObserver
from integrations.megatron_ep.routersense.trace_writer import write_json
from integrations.megatron_ep.verify_env import main as verify_env_main


def _local_expert_ids(model: torch.nn.Module) -> list[int]:
    found: set[int] = set()
    for module in model.modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is None:
            continue
        for idx in getattr(dispatcher, "local_expert_indices", []) or []:
            found.add(int(idx))
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--dispatcher", type=str, default="alltoall")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, default="native-ep-smoke")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--observer-mode", type=str, default="off")
    parser.add_argument("--scheduler-mode", type=str, default="disabled")
    parser.add_argument("--future-hint-mode", type=str, default="none")
    parser.add_argument("--control-mode", type=str, default="default_continue")
    parser.add_argument("--save-logits", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    observer_mode = validate_observer_mode(args.observer_mode)
    if args.backend != "nccl":
        payload = NativeEPSummary(
            ep_size=int(args.ep_size),
            dispatcher=str(args.dispatcher),
            backend=str(args.backend),
            status="blocked_environment",
            reason="ws2_native_ep_requires_nccl",
            details={"message": "Megatron EP smoke requires NCCL on two CUDA devices."},
        ).to_dict()
        run_dir = Path(args.output_dir) / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "summary.json", payload)
        return 2

    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    status = verify_env_main(["--model", args.model])
    if status != 0:
        payload = NativeEPSummary(
            ep_size=int(args.ep_size),
            dispatcher=str(args.dispatcher),
            backend=str(args.backend),
            status="blocked_environment",
            reason="verify_env_failed",
            details={
                "model": args.model,
                "precision": args.precision,
                "prompt_file": args.prompt_file,
            },
        ).to_dict()
        write_json(run_dir / "summary.json", payload)
        return status

    rank = 0
    local_rank = 0
    world_size = 1
    observer = RouterSenseObserver()
    policy_runtime = None
    rank_summary: dict[str, object] = {}
    try:
        ids = init_distributed(backend=args.backend, timeout_seconds=300)
        rank = ids["rank"]
        world_size = ids["world_size"]
        local_rank = ids["local_rank"]
        torch.cuda.set_device(local_rank)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        stage_barrier("device_mapping", ok=torch.cuda.current_device() == local_rank, detail=f"device={torch.cuda.current_device()} local_rank={local_rank}")
        stage_barrier("count_agreement", ok=world_size == args.ep_size, detail=f"world_size={world_size} ep_size={args.ep_size}")

        dtype = dtype_from_name(args.precision)
        prompts = load_prompts(args.prompt_file)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=Path(args.model).exists(),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"].to(device=f"cuda:{local_rank}")
        request_table_hash = hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()
        position_ids = build_position_ids(tokens)
        attention_mask = None
        stage_barrier("tokenizer", ok=True, detail=f"batch={tokens.size(0)} seqlen={tokens.size(1)}")

        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = args.ep_size
        provider.moe_token_dispatcher_type = args.dispatcher
        provider.parallel_output = False
        provider.pipeline_dtype = dtype
        provider.params_dtype = dtype
        provider.fp16 = dtype == torch.float16
        provider.bf16 = dtype == torch.bfloat16
        provider.gradient_accumulation_fusion = False
        provider.masked_softmax_fusion = False
        provider.bias_activation_fusion = False
        provider.tp_comm_overlap = False
        provider.finalize()
        stage_barrier("checkpoint_conversion", ok=True, detail=f"dispatcher={provider.moe_token_dispatcher_type}")

        models = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=True,
        )
        model = models[0].cuda(local_rank).eval()
        if observer_mode == "lightweight":
            attach_dispatch_observer(observer, rank=rank, local_rank=local_rank)(model)
        stage_barrier("observer_install", ok=True, detail=observer_mode)
        injection_config = RouterSenseInjectionConfig(
            scheduler_mode=args.scheduler_mode,
            future_hint_mode=args.future_hint_mode,
            control_mode=args.control_mode,
        )
        if injection_config.scheduler_mode != "disabled":
            policy_runtime = attach_dispatch_facade(
                model=model,
                config=injection_config,
                rank=rank,
                local_rank=local_rank,
                run_id=args.run_id,
                model_revision=args.model,
                request_table_hash=request_table_hash,
                hostname=summarize_rank_environment(rank, local_rank)["host"],
                step_id="unknown",
                microbatch_id="unknown",
                observer=observer if observer_mode == "lightweight" else None,
            )
        stage_barrier("facade_forward", ok=True, detail=injection_config.scheduler_mode)
        stage_barrier("model_load", ok=True, detail=type(model).__name__)

        with torch.inference_mode():
            logits = model(tokens, position_ids, attention_mask)
        stage_barrier("native_forward", ok=isinstance(logits, torch.Tensor), detail=str(tuple(logits.shape)))

        rows = observer.export_rows()
        policy_records = policy_runtime.export_records() if policy_runtime is not None else []
        dispatch_summary = summarize_native_dispatchers(model, rank=rank)
        observer_summary = summarize_observer_rows(rows, rank=rank) if observer_mode == "lightweight" else dispatch_summary
        local_expert_ids = _local_expert_ids(model)
        logits_path = None
        if args.save_logits:
            logits_path = run_dir / f"{args.run_id}-rank{rank}-logits.pt"
            torch.save(logits.detach().float().cpu(), logits_path)
        rank_summary = {
            **summarize_rank_environment(rank, local_rank),
            "ep_group_ranks": list(range(world_size)),
            "dispatcher_type": args.dispatcher,
            "local_expert_ids": local_expert_ids,
            "number_of_local_experts": len(local_expert_ids),
            "forward_completed": True,
            "output_checksum": float(logits.float().sum().item()),
            "output_shape": list(logits.shape),
            "trace_row_count": len(rows),
            "remote_dispatch_rows": observer_summary["remote_dispatch_rows"],
            "remote_combine_rows": observer_summary["remote_combine_rows"],
            "local_dispatch_rows": observer_summary["local_dispatch_rows"],
            "local_combine_rows": observer_summary["local_combine_rows"],
            "observer_warning_count": observer_summary.get("observer_warning_count", 0),
            "observer_phase_counts": observer_summary.get("observer_phase_counts", {}),
            "observer_mode": observer_mode,
            "scheduler_mode": injection_config.scheduler_mode,
            "future_hint_mode": injection_config.future_hint_mode,
            "control_mode": injection_config.control_mode,
            "facade_mode": "no_op_native_passthrough" if injection_config.scheduler_mode != "disabled" else "not_installed",
            "transport_mutation": False,
            "seed": args.seed,
            "precision": args.precision,
            "logits_path": str(logits_path) if logits_path is not None else None,
            "dispatcher_rows": dispatch_summary["dispatcher_rows"],
            "observer_rows": rows,
            "policy_records": policy_records,
        }
        gathered = gather_rank_payloads(rank_summary)

        if rank == 0:
            remote_dispatch_exercised = any(int(item.get("remote_dispatch_rows", 0)) > 0 for item in gathered)
            remote_combine_exercised = any(int(item.get("remote_combine_rows", 0)) > 0 for item in gathered)
            policy_name = "disabled"
            planner_ms = 0.0
            agreement_ms = 0.0
            wave_count = 0
            duplex_pair_count = 0
            p0_total_rows = 0
            p0_total_bytes = 0
            p1_total_rows = 0
            p1_total_bytes = 0
            plan_hash = None
            policy_version = injection_config.policy_version
            if injection_config.scheduler_mode != "disabled":
                for item in gathered:
                    records = item.get("policy_records", [])
                    if not records:
                        continue
                    first = records[0]
                    plan = first.get("plan", {})
                    agreement = first.get("agreement", {})
                    metrics = plan.get("metrics", {})
                    phase_coverage = metrics.get("phase_coverage", {})
                    policy_name = str(plan.get("policy_name", injection_config.scheduler_mode))
                    plan_hash = plan.get("plan_hash")
                    planner_ms = max(planner_ms, float(agreement.get("planner_ms", 0.0)))
                    agreement_ms = max(agreement_ms, float(agreement.get("agreement_ms", 0.0)))
                    wave_count = max(wave_count, int(metrics.get("wave_count", 0)))
                    duplex_pair_count = max(duplex_pair_count, int(metrics.get("duplex_pair_count", 0)))
                    p0_total_rows = max(p0_total_rows, int(phase_coverage.get("P0", {}).get("rows", 0)))
                    p0_total_bytes = max(p0_total_bytes, int(phase_coverage.get("P0", {}).get("bytes", 0)))
                    p1_total_rows = max(p1_total_rows, int(phase_coverage.get("P1", {}).get("rows", 0)))
                    p1_total_bytes = max(p1_total_bytes, int(phase_coverage.get("P1", {}).get("bytes", 0)))
                remote_dispatch_exercised = remote_dispatch_exercised or p0_total_rows > 0
                remote_combine_exercised = remote_combine_exercised or p1_total_rows > 0
            payload = NativeEPSummary(
                ep_size=args.ep_size,
                dispatcher=args.dispatcher,
                backend=args.backend,
                forward_completed=True,
                remote_dispatch_exercised=remote_dispatch_exercised,
                remote_combine_exercised=remote_combine_exercised,
                status="ready",
                reason=None,
                details={
                    "run_id": args.run_id,
                    "model": args.model,
                    "precision": args.precision,
                    "world_size": world_size,
                    "observer_mode": observer_mode,
                    "scheduler_mode": injection_config.scheduler_mode,
                    "future_hint_mode": injection_config.future_hint_mode,
                    "control_mode": injection_config.control_mode,
                    "transport_mutation": False,
                    "seed": args.seed,
                    "policy_name": policy_name,
                    "policy_version": policy_version,
                    "plan_hash": plan_hash,
                    "planner_ms": planner_ms,
                    "agreement_ms": agreement_ms,
                    "wave_count": wave_count,
                    "duplex_pair_count": duplex_pair_count,
                    "P0_total_rows": p0_total_rows,
                    "P0_total_bytes": p0_total_bytes,
                    "P1_total_rows": p1_total_rows,
                    "P1_total_bytes": p1_total_bytes,
                    "rank_summaries": gathered,
                },
            ).to_dict()
            write_json(run_dir / "summary.json", payload)
            if injection_config.scheduler_mode != "disabled":
                policy_dir = Path(args.output_dir) / "policy_shadow" / args.run_id
                policy_dir.mkdir(parents=True, exist_ok=True)
                merged = {
                    "backend": args.backend,
                    "ep_size": args.ep_size,
                    "rank_to_host": {str(item["rank"]): item["host"] for item in gathered},
                    "rank_to_device": {str(item["rank"]): item["device"] for item in gathered},
                    "policy_name": policy_name,
                    "policy_version": policy_version,
                    "future_hint_mode": injection_config.future_hint_mode,
                    "control_mode": injection_config.control_mode,
                    "transport_mutation": False,
                    "precision": args.precision,
                    "seed": args.seed,
                    "plan_hash": plan_hash,
                    "planner_ms": planner_ms,
                    "agreement_ms": agreement_ms,
                    "P0_total_rows": p0_total_rows,
                    "P0_total_bytes": p0_total_bytes,
                    "P1_total_rows": p1_total_rows,
                    "P1_total_bytes": p1_total_bytes,
                    "wave_count": wave_count,
                    "duplex_pair_count": duplex_pair_count,
                    "numerical_equivalence_passed": None,
                    "rank_summaries": gathered,
                }
                write_json(policy_dir / "merged_summary.json", merged)
                for item in gathered:
                    write_json(policy_dir / f"rank{item['rank']}_summary.json", item)
                plans_rows: list[dict[str, object]] = []
                agreement_rows: list[dict[str, object]] = []
                for item in gathered:
                    for record in item.get("policy_records", []):
                        plans_rows.append(
                            {
                                "rank": item["rank"],
                                "layer_name": record.get("layer_name"),
                                "plan": record.get("plan"),
                            }
                        )
                        agreement_rows.append(
                            {
                                "rank": item["rank"],
                                "layer_name": record.get("layer_name"),
                                "agreement": record.get("agreement"),
                                "decision": record.get("decision"),
                            }
                        )
                from integrations.megatron_ep.routersense.trace_writer import write_jsonl
                write_jsonl(policy_dir / "plans.jsonl", plans_rows)
                write_jsonl(policy_dir / "agreement.jsonl", agreement_rows)
        stage_barrier("artifact_flush", ok=True, detail=str(run_dir))
        return 0
    except Exception as exc:
        error = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "backend": args.backend,
            "model": args.model,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / f"{args.run_id}-rank{rank}-error.json", error)
        if rank == 0:
            payload = NativeEPSummary(
                ep_size=args.ep_size,
                dispatcher=args.dispatcher,
                backend=args.backend,
                status="runtime_failure",
                reason=f"{type(exc).__name__}: {exc}",
                details=error,
            ).to_dict()
            write_json(run_dir / "summary.json", payload)
        return 1
    finally:
        destroy_distributed()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    raise SystemExit(main())
