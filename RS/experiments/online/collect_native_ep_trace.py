#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from rs.runtime.online.megatron_ep.host import (
    attach_dispatch_facade,
    attach_dispatch_observer,
    build_position_ids,
    destroy_distributed,
    dtype_from_name,
    gather_rank_payloads,
    init_distributed,
    load_prompts,
    stage_barrier,
    summarize_observer_rows,
    summarize_rank_environment,
    validate_observer_mode,
)
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.observer import RouterSenseObserver
from rs.runtime.online.megatron_ep.trace_writer import write_json, write_jsonl
from experiments.online._verify_env import main as verify_env_main


def _result_exit_code(payload: dict[str, object]) -> int:
    execution_mode = str(payload.get("execution_mode", ""))
    correctness_status = str(payload.get("correctness_status", ""))
    if execution_mode in {"online_ws2_route_partition_only", "online_ws2_hidden_dispatch_only"}:
        return 0 if correctness_status == "metadata_passed" else 2
    if execution_mode == "online_ws2_native_ep_moe_layer_harness":
        return 0 if correctness_status in {"passed", "skipped_no_remote_route"} else 2
    return 0 if payload.get("numerical_correctness_pass") is True else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--dispatcher", type=str, default="alltoall")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--prompt-file", type=str, default=str(Path(__file__).resolve().parent / "prompts.json"))
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, default="native-ep-trace")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--observer-mode", type=str, default="lightweight")
    parser.add_argument("--scheduler-mode", type=str, default="disabled")
    parser.add_argument("--future-hint-mode", type=str, default="none")
    parser.add_argument("--control-mode", type=str, default="default_continue")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    observer_mode = validate_observer_mode(args.observer_mode)
    if args.backend != "nccl":
        run_dir = Path(args.output_dir) / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            run_dir / "summary.json",
            {
                "pipeline": "host_runtime_native_ep",
                "host_runtime": "megatron_core",
                "status": "blocked_environment",
                "reason": "ws2_native_ep_requires_nccl",
                "future_hint_mode": "none",
                "facade_mode": "not_started",
            },
        )
        return 2

    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status = verify_env_main(["--model", args.model])
    if status != 0:
        write_json(
            run_dir / "summary.json",
            {
                "pipeline": "host_runtime_native_ep",
                "host_runtime": "megatron_core",
                "status": "blocked_environment",
                "reason": "verify_env_failed",
                "future_hint_mode": "none",
                "facade_mode": "not_started",
            },
        )
        write_jsonl(run_dir / "trace.jsonl", [])
        return status

    rank = 0
    local_rank = 0
    world_size = 1
    observer = RouterSenseObserver()
    policy_runtime = None
    try:
        ids = init_distributed(backend=args.backend, timeout_seconds=300)
        rank = ids["rank"]
        world_size = ids["world_size"]
        local_rank = ids["local_rank"]
        torch.cuda.set_device(local_rank)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        prompts = load_prompts(args.prompt_file)
        dtype = dtype_from_name(args.precision)
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
            logits = model(tokens, position_ids, None)
        stage_barrier("observer_forward", ok=isinstance(logits, torch.Tensor), detail=str(tuple(logits.shape)))

        rows = observer.export_rows()
        policy_records = policy_runtime.export_records() if policy_runtime is not None else []
        observer_summary = summarize_observer_rows(rows, rank=rank)
        rank_summary = {
            **summarize_rank_environment(rank, local_rank),
            "run_id": args.run_id,
            "pipeline": "host_runtime_native_ep",
            "host_runtime": "megatron_core",
            "future_hint_mode": injection_config.future_hint_mode,
            "control_mode": injection_config.control_mode,
            "facade_mode": "no_op_native_passthrough" if injection_config.scheduler_mode != "disabled" else "not_installed",
            "observer_mode": observer_mode,
            "scheduler_mode": injection_config.scheduler_mode,
            "model": args.model,
            "dispatcher": args.dispatcher,
            "precision": args.precision,
            "seed": args.seed,
            "transport_mutation": False,
            "trace_row_count": len(rows),
            "output_checksum": float(logits.float().sum().item()),
            "remote_dispatch_rows": observer_summary["remote_dispatch_rows"],
            "remote_combine_rows": observer_summary["remote_combine_rows"],
            "local_dispatch_rows": observer_summary["local_dispatch_rows"],
            "local_combine_rows": observer_summary["local_combine_rows"],
            "observer_warning_count": observer_summary["observer_warning_count"],
            "observer_phase_counts": observer_summary["observer_phase_counts"],
            "policy_records": policy_records,
        }
        trace_path = run_dir / f"{args.run_id}-rank{rank}.jsonl"
        metadata_path = run_dir / f"{args.run_id}-rank{rank}_metadata.json"
        summary_path = run_dir / f"{args.run_id}-rank{rank}_summary.json"
        write_jsonl(trace_path, rows)
        write_json(metadata_path, rank_summary)
        write_json(summary_path, rank_summary)

        gathered = gather_rank_payloads(
            {
                "rank_summary": rank_summary,
                "trace_path": str(trace_path),
                "metadata_path": str(metadata_path),
                "summary_path": str(summary_path),
            }
        )
        if rank == 0:
            write_json(
                run_dir / f"{args.run_id}-merged.json",
                {
                    "world_size": world_size,
                    "backend": args.backend,
                    "dispatcher": args.dispatcher,
                    "pipeline": "host_runtime_native_ep",
                    "host_runtime": "megatron_core",
                    "future_hint_mode": injection_config.future_hint_mode,
                    "control_mode": injection_config.control_mode,
                    "facade_mode": "no_op_native_passthrough" if injection_config.scheduler_mode == "native_order" else "not_installed",
                    "observer_mode": observer_mode,
                    "scheduler_mode": injection_config.scheduler_mode,
                    "rank_artifacts": gathered,
                },
            )
        stage_barrier("artifact_flush", ok=True, detail=str(run_dir))
        return 0
    except Exception as exc:
        write_json(
            run_dir / f"{args.run_id}-rank{rank}-error.json",
            {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "backend": args.backend,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    finally:
        destroy_distributed()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    raise SystemExit(main())
