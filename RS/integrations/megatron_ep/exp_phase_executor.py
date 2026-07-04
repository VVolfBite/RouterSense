#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.megatron_ep.native_runtime import (
    attach_dispatch_facade,
    build_position_ids,
    destroy_distributed,
    dtype_from_name,
    gather_rank_payloads,
    init_distributed,
    load_prompts,
    stage_barrier,
    summarize_native_dispatchers,
    summarize_rank_environment,
)
from integrations.megatron_ep.routersense.contracts import NativeEPSummary, RouterSenseInjectionConfig
from integrations.megatron_ep.routersense.dispatcher_facade import SelectedLayerStop
from integrations.megatron_ep.routersense.policy.registry import resolve_phase_policy, supported_phase_policies
from integrations.megatron_ep.routersense.trace_writer import write_json, write_jsonl
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


def _source_provenance() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_available": bool(torch.distributed.is_nccl_available()),
        "entrypoint": str(Path(__file__).resolve()),
    }


def _selector_matches(selector: str, value: str) -> bool:
    if selector in {"", "all", "both"}:
        return True
    selected = {item.strip() for item in selector.split(",") if item.strip()}
    return value in selected


def _capture_enabled(*, layer_selector: str, phase_selector: str, layer_id: str, phase: str) -> bool:
    return _selector_matches(layer_selector, layer_id) and _selector_matches(phase_selector, phase)


def _phase_context_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {
        "p0_remote_rows": 0,
        "p1_remote_rows": 0,
        "p0_remote_bytes_hidden": 0,
        "p0_remote_bytes_probs": 0,
        "p1_remote_bytes_hidden": 0,
        "p0_remote_flow_count": 0,
        "p1_remote_flow_count": 0,
    }
    for row in rows:
        phase = str(row.get("phase"))
        for bundle in row.get("transport_bundles", []) or []:
            segment = bundle.get("outgoing_segment", {})
            if bool(segment.get("is_local", False)):
                continue
            row_count = int(segment.get("row_count", 0))
            stats["p0_remote_rows" if phase == "P0" else "p1_remote_rows"] += row_count
            stats["p0_remote_flow_count" if phase == "P0" else "p1_remote_flow_count"] += 1
            for payload in bundle.get("payload_slices", []) or []:
                role = str(payload.get("tensor_role"))
                byte_count = int(payload.get("payload_byte_count", 0))
                if phase == "P0" and role == "hidden_states":
                    stats["p0_remote_bytes_hidden"] += byte_count
                elif phase == "P0" and role == "routing_probs":
                    stats["p0_remote_bytes_probs"] += byte_count
                elif phase == "P1" and role == "hidden_states":
                    stats["p1_remote_bytes_hidden"] += byte_count
    return stats


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    cosine = torch.nn.functional.cosine_similarity(af.reshape(1, -1), bf.reshape(1, -1)).item()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": float(cosine),
    }


def _write_not_triggered(path: Path) -> None:
    write_json(path, {"status": "not_triggered"})


def _effective_policy_name(policy: str, scheduler_mode: str) -> str:
    if policy:
        return str(policy)
    if scheduler_mode in set(supported_phase_policies()):
        return str(scheduler_mode)
    return ""


def _failure_report(
    *,
    stage: str,
    exc: BaseException,
    rank: int,
    local_rank: int,
    plan_hash: str | None = None,
    layer_id: str | None = None,
    phase: str | None = None,
    wave_id: int | None = None,
    bucket_id: str | None = None,
    tensor_role: str | None = None,
    expected_shape: list[int] | None = None,
    actual_shape: list[int] | None = None,
    expected_dtype: str | None = None,
    actual_dtype: str | None = None,
    expected_splits: list[int] | None = None,
    actual_splits: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "first_failure_stage": stage,
        "forward_epoch": 0,
        "layer_id": layer_id,
        "phase": phase,
        "rank": rank,
        "local_rank": local_rank,
        "wave_id": wave_id,
        "bucket_id": bucket_id,
        "tensor_role": tensor_role,
        "expected_shape": expected_shape,
        "actual_shape": actual_shape,
        "expected_dtype": expected_dtype,
        "actual_dtype": actual_dtype,
        "expected_splits": expected_splits,
        "actual_splits": actual_splits,
        "plan_hash": plan_hash,
        "exception_summary": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def _write_rank_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    rank: int,
    logits: torch.Tensor | None,
    runtime: Any | None,
    native_dispatch_summary: dict[str, Any],
    rank_summary: dict[str, Any],
    save_logits: bool,
    capture_layer_selector: str,
    capture_phase_selector: str,
) -> dict[str, Any]:
    logits_path = None
    if save_logits and logits is not None:
        logits_path = run_dir / f"{run_id}-rank{rank}-logits.pt"
        torch.save(logits.detach().float().cpu(), logits_path)
        rank_summary["logits_path"] = str(logits_path)
    write_json(run_dir / f"rank{rank}_summary.json", rank_summary)
    write_json(run_dir / f"rank{rank}_native_dispatch.json", native_dispatch_summary)
    if runtime is not None:
        write_jsonl(run_dir / f"rank{rank}_control_timeline.jsonl", runtime.export_control_timeline())
        write_jsonl(run_dir / f"rank{rank}_control_commands.jsonl", runtime.export_control_commands())
        write_json(run_dir / f"rank{rank}_assertions.json", runtime.export_assertions())
        write_jsonl(run_dir / f"rank{rank}_phase_contexts.jsonl", runtime.export_phase_contexts())
        write_jsonl(run_dir / f"rank{rank}_transport_bundles.jsonl", runtime.export_transport_bundles())
        write_jsonl(run_dir / f"rank{rank}_scheduled_phase_plans.jsonl", runtime.export_scheduled_phase_plans())
        transport_results = []
        adapter = getattr(runtime, "transport_adapter", None)
        if adapter is not None:
            transport_results = adapter.export_results()
        else:
            transport_results = runtime.export_transport_execution_results()
        write_jsonl(run_dir / f"rank{rank}_transport_execution.jsonl", transport_results)
        write_jsonl(run_dir / f"rank{rank}_captured_phase_tensors.jsonl", runtime.export_captured_phase_tensors())
        capture_dir = run_dir / "captured_phase_tensors"
        capture_dir.mkdir(parents=True, exist_ok=True)
        for item in runtime.captured_phase_tensors:
            layer_id = str(item["layer_id"])
            phase = str(item["phase"])
            if not _capture_enabled(
                layer_selector=capture_layer_selector,
                phase_selector=capture_phase_selector,
                layer_id=layer_id,
                phase=phase,
            ):
                continue
            tensor_path = capture_dir / f"rank{rank}_layer{layer_id}_{phase}_{item['tensor_role']}.pt"
            torch.save(item["tensor"], tensor_path)
    return rank_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--dispatcher", type=str, default="alltoall")
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, default="phase-executor")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--policy", type=str, default="")
    parser.add_argument("--scheduler-mode", type=str, default="disabled")
    parser.add_argument("--execution-mode", type=str, default="native_passthrough")
    parser.add_argument("--control-mode", type=str, default="sync_before_phase")
    parser.add_argument("--bucket-rows", type=int, default=0)
    parser.add_argument("--p0-weight", type=float, default=1.0)
    parser.add_argument("--p1-reservation-weight", type=float, default=1.0)
    parser.add_argument("--p2-hint-weight", type=float, default=1.0)
    parser.add_argument("--p2-hint-artifact", type=str, default="")
    parser.add_argument("--schedule-layer-selector", type=str, default="all")
    parser.add_argument("--schedule-phase-selector", type=str, default="both")
    parser.add_argument("--capture-layer-selector", type=str, default="all")
    parser.add_argument("--capture-phase-selector", type=str, default="both")
    parser.add_argument("--capture-phase-tensors", action="store_true", default=False)
    parser.add_argument("--stop-after-selected-layer", action="store_true", default=False)
    parser.add_argument("--p2-hint-mode", type=str, default="none")
    parser.add_argument("--executor-heartbeat-path", type=str, default="")
    parser.add_argument("--executor-phase-timeout-sec", type=int, default=120)
    parser.add_argument("--save-logits", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "source_provenance.json", _source_provenance())
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    if args.backend != "nccl":
        payload = NativeEPSummary(
            ep_size=int(args.ep_size),
            dispatcher=str(args.dispatcher),
            backend=str(args.backend),
            status="blocked_environment",
            reason="phase_executor_requires_nccl",
            details={"execution_mode": args.execution_mode},
        ).to_dict()
        write_json(run_dir / "summary.json", payload)
        return 2

    status = verify_env_main(["--model", args.model])
    if status != 0:
        payload = NativeEPSummary(
            ep_size=int(args.ep_size),
            dispatcher=str(args.dispatcher),
            backend=str(args.backend),
            status="blocked_environment",
            reason="verify_env_failed",
            details={"model": args.model},
        ).to_dict()
        write_json(run_dir / "summary.json", payload)
        return status

    rank = 0
    local_rank = 0
    world_size = 1
    runtime = None
    logits = None
    try:
        ids = init_distributed(backend=args.backend, timeout_seconds=300)
        rank = ids["rank"]
        world_size = ids["world_size"]
        local_rank = ids["local_rank"]
        torch.cuda.set_device(local_rank)
        stdout_log = (run_dir / f"stdout-rank{rank}.log").open("w", encoding="utf-8")
        stderr_log = (run_dir / f"stderr-rank{rank}.log").open("w", encoding="utf-8")
        sys.stdout = stdout_log
        sys.stderr = stderr_log
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        stage_barrier("device_mapping", ok=torch.cuda.current_device() == local_rank, detail=f"device={torch.cuda.current_device()} local_rank={local_rank}")
        stage_barrier("count_agreement", ok=world_size == args.ep_size, detail=f"world_size={world_size} ep_size={args.ep_size}")

        dtype = dtype_from_name(args.precision)
        if args.p2_hint_artifact:
            os.environ["ROUTERSENSE_P2_HINT_ARTIFACT"] = args.p2_hint_artifact
        prompts = load_prompts(args.prompt_file)
        from transformers import AutoTokenizer
        from megatron.bridge import AutoBridge

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=Path(args.model).exists(),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        tokens = encoded["input_ids"].to(device=f"cuda:{local_rank}")
        position_ids = build_position_ids(tokens)
        attention_mask = None
        request_table_hash = hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()
        stage_barrier("tokenizer", ok=True, detail=f"batch={tokens.size(0)} seqlen={tokens.size(1)}")

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

        models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
        model = models[0].cuda(local_rank).eval()
        effective_policy_name = _effective_policy_name(args.policy, args.scheduler_mode)
        policy_capabilities = (
            resolve_phase_policy(
                policy_name=effective_policy_name,
                bucket_rows=args.bucket_rows,
                p0_weight=args.p0_weight,
                p1_reservation_weight=args.p1_reservation_weight,
                p2_hint_weight=args.p2_hint_weight,
                p2_hint_artifact=args.p2_hint_artifact,
            ).capabilities.to_dict()
            if effective_policy_name
            else None
        )
        injection_config = RouterSenseInjectionConfig(
            policy=effective_policy_name,
            scheduler_mode=args.scheduler_mode,
            execution_mode=args.execution_mode,
            future_hint_mode="none",
            p2_hint_mode=args.p2_hint_mode,
            control_mode=args.control_mode,
            bucket_rows=args.bucket_rows,
            p0_weight=args.p0_weight,
            p1_reservation_weight=args.p1_reservation_weight,
            p2_hint_weight=args.p2_hint_weight,
            p2_hint_artifact=args.p2_hint_artifact,
            schedule_layer_selector=args.schedule_layer_selector,
            schedule_phase_selector=args.schedule_phase_selector,
            capture_phase_tensors=args.capture_phase_tensors,
            stop_after_selected_layer=args.stop_after_selected_layer,
            executor_heartbeat_path=args.executor_heartbeat_path or str(run_dir),
            executor_phase_timeout_sec=args.executor_phase_timeout_sec,
        )
        if injection_config.scheduler_mode != "disabled" or injection_config.policy or injection_config.capture_phase_tensors:
            runtime = attach_dispatch_facade(
                model=model,
                config=injection_config,
                rank=rank,
                local_rank=local_rank,
                run_id=args.run_id,
                model_revision=args.model,
                request_table_hash=request_table_hash,
                hostname=summarize_rank_environment(rank, local_rank)["host"],
                step_id="forward-0",
                microbatch_id="mb-0",
                observer=None,
            )
        stage_barrier("model_load", ok=True, detail=type(model).__name__)

        partial_stop = False
        try:
            with torch.inference_mode():
                logits = model(tokens, position_ids, attention_mask)
        except SelectedLayerStop:
            partial_stop = True
            logits = None
        stage_barrier("native_forward", ok=bool(partial_stop or isinstance(logits, torch.Tensor)), detail="partial-stop" if partial_stop else str(tuple(logits.shape)))

        native_dispatch_summary = summarize_native_dispatchers(model, rank=rank)
        local_expert_ids = _local_expert_ids(model)
        transport_results = []
        phase_context_rows: list[dict[str, Any]] = []
        if runtime is not None:
            adapter = getattr(runtime, "transport_adapter", None)
            if adapter is not None:
                transport_results = adapter.export_results()
            else:
                transport_results = runtime.export_transport_execution_results()
            phase_context_rows = runtime.export_phase_contexts()
        phase_stats = _phase_context_stats(phase_context_rows)
        rank_summary = {
            **summarize_rank_environment(rank, local_rank),
            "ep_group_ranks": list(range(world_size)),
            "dispatcher_type": args.dispatcher,
            "local_expert_ids": local_expert_ids,
            "number_of_local_experts": len(local_expert_ids),
            "forward_completed": bool(logits is not None),
            "forward_partial_stop": partial_stop,
            "output_checksum": float(logits.float().sum().item()) if isinstance(logits, torch.Tensor) else None,
            "output_shape": list(logits.shape) if isinstance(logits, torch.Tensor) else None,
            "remote_dispatch_rows": phase_stats["p0_remote_rows"],
            "remote_combine_rows": phase_stats["p1_remote_rows"],
            "local_dispatch_rows": native_dispatch_summary["local_dispatch_rows"],
            "local_combine_rows": native_dispatch_summary["local_combine_rows"],
            **phase_stats,
            "policy_name": effective_policy_name or "disabled",
            "policy_version": injection_config.policy_version if effective_policy_name else "",
            "policy_capabilities": policy_capabilities,
            "legacy_scheduler_mode": injection_config.scheduler_mode,
            "execution_mode": injection_config.execution_mode,
            "control_mode": injection_config.control_mode,
            "bucket_rows": injection_config.bucket_rows,
            "p0_weight": injection_config.p0_weight,
            "p1_reservation_weight": injection_config.p1_reservation_weight,
            "p2_hint_weight": injection_config.p2_hint_weight,
            "p2_hint_mode": injection_config.p2_hint_mode,
            "schedule_layer_selector": injection_config.schedule_layer_selector,
            "schedule_phase_selector": injection_config.schedule_phase_selector,
            "seed": args.seed,
            "precision": args.precision,
            "transport_mutation": bool(
                effective_policy_name
                and injection_config.execution_mode == "phase_sync_wave"
            ),
            "dispatcher_rows": native_dispatch_summary["dispatcher_rows"],
            "transport_execution_count": len(transport_results),
        }
        rank_summary = _write_rank_artifacts(
            run_dir=run_dir,
            run_id=args.run_id,
            rank=rank,
            logits=logits,
            runtime=runtime,
            native_dispatch_summary=native_dispatch_summary,
            rank_summary=rank_summary,
            save_logits=args.save_logits,
            capture_layer_selector=args.capture_layer_selector,
            capture_phase_selector=args.capture_phase_selector,
        )
        gathered = gather_rank_payloads(rank_summary)

        if rank == 0:
            remote_dispatch_exercised = any(int(item.get("remote_dispatch_rows", 0)) > 0 for item in gathered)
            remote_combine_exercised = any(int(item.get("remote_combine_rows", 0)) > 0 for item in gathered)
            transport_mutation = bool(
                effective_policy_name
                and injection_config.execution_mode == "phase_sync_wave"
            )
            details = {
                "run_id": args.run_id,
                "model": args.model,
                "precision": args.precision,
                "seed": args.seed,
                "world_size": world_size,
                "policy_name": effective_policy_name or "disabled",
                "policy_version": injection_config.policy_version if effective_policy_name else "",
                "policy_capabilities": policy_capabilities,
                "legacy_scheduler_mode": injection_config.scheduler_mode,
                "execution_mode": injection_config.execution_mode,
                "control_mode": injection_config.control_mode,
                "bucket_rows": injection_config.bucket_rows,
                "p0_weight": injection_config.p0_weight,
                "p1_reservation_weight": injection_config.p1_reservation_weight,
                "p2_hint_weight": injection_config.p2_hint_weight,
                "p2_hint_mode": injection_config.p2_hint_mode,
                "schedule_layer_selector": injection_config.schedule_layer_selector,
                "schedule_phase_selector": injection_config.schedule_phase_selector,
                "transport_mutation": transport_mutation,
                "rank_summaries": gathered,
            }
            if runtime is not None:
                details["scheduled_phase_plans_path_hint"] = str(run_dir / "rank0_scheduled_phase_plans.jsonl")
            payload = NativeEPSummary(
                ep_size=args.ep_size,
                dispatcher=args.dispatcher,
                backend=args.backend,
                forward_completed=all(bool(item.get("forward_completed") or item.get("forward_partial_stop")) for item in gathered),
                remote_dispatch_exercised=remote_dispatch_exercised,
                remote_combine_exercised=remote_combine_exercised,
                status="ready",
                reason=None,
                details=details,
            ).to_dict()
            write_json(run_dir / "summary.json", payload)
            _write_not_triggered(run_dir / "failure_report.json")
            _write_not_triggered(run_dir / "watchdog_report.json")
        stage_barrier("artifact_flush", ok=True, detail=str(run_dir))
        return 0
    except Exception as exc:
        failure = _failure_report(
            stage="phase_executor_runtime",
            exc=exc,
            rank=rank,
            local_rank=local_rank,
        )
        write_json(run_dir / "failure_report.json", failure)
        if rank == 0:
            payload = NativeEPSummary(
                ep_size=args.ep_size,
                dispatcher=args.dispatcher,
                backend=args.backend,
                status="runtime_failure",
                reason=f"{type(exc).__name__}: {exc}",
                details=failure,
            ).to_dict()
            write_json(run_dir / "summary.json", payload)
        return 1
    finally:
        try:
            if runtime is not None and hasattr(runtime, "original_all_to_all"):
                import megatron.core.transformer.moe.token_dispatcher as token_dispatcher_mod
                token_dispatcher_mod.all_to_all = runtime.original_all_to_all
        except Exception:
            pass
        destroy_distributed()


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    raise SystemExit(main())
