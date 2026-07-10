#!/usr/bin/env python3
"""Formal native Megatron EP observation entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
import traceback
from pathlib import Path

import torch

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import write_json, write_jsonl
from rs.core.experiment_config import RunConfig, load_run_config
from rs.runtime.online.megatron_ep.contracts import (
    ExecutionSelection,
    OnlinePolicyParameters,
    OnlineRuntimeConfig,
    OnlineValidationConfig,
)
from rs.runtime.online.megatron_ep.host import (
    attach_dispatch_facade,
    attach_formal_online_runtime,
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
)
from rs.runtime.online.megatron_ep.observation import RouterSenseObserver

from experiments.online.support.environment_validation import main as verify_env_main


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def _resolve_model_path(config: RunConfig) -> str:
    return config.model.local_path or config.model.model_id


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_run_config(
        config_path=args.config,
        overrides=list(args.override),
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    model_path = _resolve_model_path(config)
    run_id = config.run.name
    run_dir = Path(config.artifact.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "run_kind": config.run.kind,
            "policy_name": "disabled",
            "transport_mutation": False,
            "model_id": config.model.model_id,
            "model_path": model_path,
            "ep_size": config.topology.ep_size,
            "precision": config.runtime.precision,
            "prompt_file": config.workload.prompts,
            "observation_profile": config.observation.profile,
            "source_config_path": config.source_config_path,
        },
    )

    status = verify_env_main(["--model", model_path])
    if status != 0:
        write_json(
            run_dir / "summary.json",
            {
                "pipeline": "host_runtime_native_ep",
                "host_runtime": "megatron_core",
                "status": "blocked_environment",
                "reason": "verify_env_failed",
                "transport_mutation": False,
                "policy_name": "disabled",
            },
        )
        write_jsonl(run_dir / "trace.jsonl", [])
        return status

    rank = 0
    local_rank = 0
    world_size = 1
    observer = RouterSenseObserver()
    policy_runtime = None
    experiment_timeline: list[dict[str, object]] = []

    def record_event(event: str, **detail: object) -> None:
        experiment_timeline.append(
            {
                "event": event,
                "ts_us": int(time.time() * 1e6),
                "monotonic_ns": time.monotonic_ns(),
                "rank": rank,
                "local_rank": local_rank,
                **detail,
            }
        )
    try:
        record_event("main_enter")
        ids = init_distributed(backend="nccl", timeout_seconds=300)
        rank = ids["rank"]
        world_size = ids["world_size"]
        local_rank = ids["local_rank"]
        record_event("distributed_initialized", world_size=world_size)
        torch.cuda.set_device(local_rank)
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        prompts_start_ns = time.monotonic_ns()
        prompts = load_prompts(config.workload.prompts)
        prompts_end_ns = time.monotonic_ns()
        record_event("prompts_loaded", prompt_count=len(prompts), duration_us=(prompts_end_ns - prompts_start_ns) / 1000.0)
        dtype = dtype_from_name(config.runtime.precision)
        from transformers import AutoTokenizer
        from megatron.bridge import AutoBridge

        tokenizer_start_ns = time.monotonic_ns()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=config.model.trust_remote_code,
            local_files_only=Path(model_path).exists(),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        tokenizer_end_ns = time.monotonic_ns()
        record_event("tokenizer_and_encode_done", duration_us=(tokenizer_end_ns - tokenizer_start_ns) / 1000.0, batch_rows=int(encoded["input_ids"].shape[0]), seq_len=int(encoded["input_ids"].shape[1]))
        tokens = encoded["input_ids"].to(device=f"cuda:{local_rank}")
        request_table_hash = hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()
        position_ids = build_position_ids(tokens)

        model_load_start_ns = time.monotonic_ns()
        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=config.model.trust_remote_code)
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.expert_model_parallel_size = config.topology.ep_size
        provider.moe_token_dispatcher_type = config.runtime.dispatcher
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
        models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
        model = models[0].cuda(local_rank).eval()
        model_load_end_ns = time.monotonic_ns()
        record_event("model_loaded", duration_us=(model_load_end_ns - model_load_start_ns) / 1000.0)

        attach_dispatch_observer(
            observer,
            rank=rank,
            local_rank=local_rank,
            include_tensor_values=(
                config.observation.profile == "debug" and bool(config.observation.capture_enabled)
            ),
        )(model)
        if config.observation.profile in {"perf", "execution", "debug"}:
            policy_runtime = attach_formal_online_runtime(
                model=model,
                runtime_config=OnlineRuntimeConfig(
                    policy_name="disabled",
                    execution_mode="native_passthrough",
                    control_mode="none",
                    execution_selection=ExecutionSelection(layer_selector="all", phase_selector="both", bucket_rows=0),
                    policy_parameters=OnlinePolicyParameters(),
                    observation=config.observation.to_dict(),
                    validation=OnlineValidationConfig(),
                ),
                rank=rank,
                local_rank=local_rank,
                run_id=run_id,
                model_revision=model_path,
                request_table_hash=request_table_hash,
                hostname=summarize_rank_environment(rank, local_rank)["host"],
                step_id="forward-0",
                microbatch_id="mb-0",
                observer=observer,
            )
        stage_barrier("model_load", ok=True, detail=type(model).__name__)

        forward_start_ns = time.monotonic_ns()
        record_event("forward_start")
        policy_runtime.begin_forward(forward_epoch=0)
        with torch.inference_mode():
            logits = model(tokens, position_ids, None)
        policy_runtime.end_forward()
        forward_end_ns = time.monotonic_ns()
        record_event("forward_end", total_forward_us=(forward_end_ns - forward_start_ns) / 1000.0)
        stage_barrier("observer_forward", ok=isinstance(logits, torch.Tensor), detail=str(tuple(logits.shape)))

        rows = observer.export_rows()
        observer_summary = summarize_observer_rows(rows, rank=rank)
        rank_summary = {
            **summarize_rank_environment(rank, local_rank),
            "run_id": run_id,
            "pipeline": "host_runtime_native_ep",
            "host_runtime": "megatron_core",
            "control_mode": "none",
            "policy_name": "disabled",
            "execution_mode": "native_passthrough",
            "transport_mutation": False,
            "observer_mode": config.observation.profile,
            "model": model_path,
            "dispatcher": config.runtime.dispatcher,
            "precision": config.runtime.precision,
            "trace_row_count": len(rows),
            "output_checksum": float(logits.float().sum().item()),
            "remote_dispatch_rows": observer_summary["remote_dispatch_rows"],
            "remote_combine_rows": observer_summary["remote_combine_rows"],
            "local_dispatch_rows": observer_summary["local_dispatch_rows"],
            "local_combine_rows": observer_summary["local_combine_rows"],
            "observer_warning_count": observer_summary["observer_warning_count"],
            "observer_phase_counts": observer_summary["observer_phase_counts"],
        }
        write_jsonl(run_dir / f"rank{rank}_observer.jsonl", rows)
        write_json(run_dir / f"rank{rank}_native_dispatch.json", rank_summary)
        write_json(run_dir / f"rank{rank}_experiment_timeline.json", {"events": experiment_timeline})
        if policy_runtime is not None:
            write_jsonl(run_dir / f"rank{rank}_phase_contexts.jsonl", policy_runtime.export_phase_contexts())
            write_jsonl(run_dir / f"rank{rank}_transport_bundles.jsonl", policy_runtime.export_transport_bundles())
        gathered = gather_rank_payloads(rank_summary)
        if rank == 0:
            payload = {
                "pipeline": "host_runtime_native_ep",
                "host_runtime": "megatron_core",
                "status": "ready",
                "policy_name": "disabled",
                "transport_mutation": False,
                "total_forward_us": next((float(item.get("total_forward_us")) for item in reversed(experiment_timeline) if "total_forward_us" in item), 0.0),
                "remote_dispatch_exercised": any(int(item.get("remote_dispatch_rows", 0)) > 0 for item in gathered),
                "remote_combine_exercised": any(int(item.get("remote_combine_rows", 0)) > 0 for item in gathered),
                "rank_summaries": gathered,
            }
            write_json(run_dir / "summary.json", payload)
        return 0
    except Exception as exc:
        record_event("runtime_exception", exception=f"{type(exc).__name__}: {exc}")
        write_json(run_dir / f"rank{rank}_experiment_timeline.json", {"events": experiment_timeline})
        write_json(
            run_dir / "summary.json",
            {
                "pipeline": "host_runtime_native_ep",
                "host_runtime": "megatron_core",
                "status": "runtime_failure",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    finally:
        destroy_distributed()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    raise SystemExit(main())
