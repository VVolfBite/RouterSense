#!/usr/bin/env python3
"""Formal online policy-correctness entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import write_json
from rs.core.experiment_config import RunConfig, load_run_config
from rs.runtime.online.megatron_ep.contracts import (
    ExecutionSelection,
    NativeEPSummary,
    OnlinePolicyParameters,
    OnlineRuntimeConfig,
    OnlineValidationConfig,
)
from rs.runtime.online.megatron_ep.execution.audit import ExecutionAuditInput, build_execution_audit
from rs.runtime.online.megatron_ep.host import (
    attach_formal_online_runtime,
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
from rs.runtime.online.megatron_ep.runtime import SelectedLayerStop
from rs.scheduling.registry import resolve_phase_policy

from experiments.online.support.environment_validation import main as verify_env_main
from experiments.online.support.phase_executor_artifacts import (
    effective_policy_name as resolve_effective_policy_name,
    failure_report,
    local_expert_ids as collect_local_expert_ids,
    phase_context_stats,
    source_provenance,
    write_not_triggered,
    write_rank_artifacts,
)

_failure_report = failure_report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def _resolve_model_path(config: RunConfig) -> str:
    return config.model.local_path or config.model.model_id


def _build_online_runtime_config(config: RunConfig) -> OnlineRuntimeConfig:
    return OnlineRuntimeConfig(
        policy_name=config.online_policy.name,
        execution_mode=config.execution.mode,
        control_mode=config.runtime.control_mode,
        execution_selection=ExecutionSelection(
            layer_selector=config.execution.schedule.layer_selector,
            phase_selector=config.execution.schedule.phase_selector,
            bucket_rows=config.execution.bucket_rows,
        ),
        policy_parameters=OnlinePolicyParameters(
            p0_weight=config.online_policy.parameters.p0_weight,
            p1_reservation_weight=config.online_policy.parameters.p1_reservation_weight,
            p2_hint_weight=config.online_policy.parameters.p2_hint_weight,
            p2_hint_mode=config.online_policy.p2.mode,
            p2_hint_artifact=config.online_policy.p2.artifact,
            calibrated_p2_enabled=config.online_policy.p2.mode == "calibrated_artifact",
        ),
        observation=config.observation.to_dict(),
        validation=OnlineValidationConfig(
            stop_after_selected_layer=bool(config.validation.stop_after_selected_layer),
            executor_heartbeat_path=str(config.artifact.output_root),
            executor_phase_timeout_sec=120,
        ),
    )


def _policy_capabilities(config: RunConfig) -> dict[str, Any] | None:
    policy_name = resolve_effective_policy_name(config.online_policy.name, "disabled")
    if not policy_name:
        return None
    return resolve_phase_policy(
        policy_name=policy_name,
        bucket_rows=config.execution.bucket_rows,
        p0_weight=config.online_policy.parameters.p0_weight,
        p1_reservation_weight=config.online_policy.parameters.p1_reservation_weight,
        p2_hint_weight=config.online_policy.parameters.p2_hint_weight,
        p2_hint_artifact=config.online_policy.p2.artifact,
    ).capabilities.to_dict()


def _build_run_manifest(config: RunConfig, *, run_id: str, model_path: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_kind": config.run.kind,
        "policy_name": config.online_policy.name,
        "execution_mode": config.execution.mode,
        "control_mode": config.runtime.control_mode,
        "observation_profile": config.observation.profile,
        "model_id": config.model.model_id,
        "model_path": model_path,
        "precision": config.runtime.precision,
        "ep_size": config.topology.ep_size,
        "prompt_file": config.workload.prompts,
        "bucket_rows": config.execution.bucket_rows,
        "schedule_layer_selector": config.execution.schedule.layer_selector,
        "schedule_phase_selector": config.execution.schedule.phase_selector,
        "capture_layer_selector": config.observation.capture_layer_selector,
        "capture_phase_selector": config.observation.capture_phase_selector,
        "stop_after_selected_layer": config.validation.stop_after_selected_layer,
        "save_logits": config.validation.save_logits,
        "source_config_path": config.source_config_path,
    }


def _build_rank_audits(
    *,
    runtime: Any | None,
    transport_results: list[dict[str, Any]],
    policy_enabled: bool,
) -> dict[str, Any]:
    if runtime is None:
        return {"status": "not_applicable", "audits": []}
    plans = runtime.export_scheduled_phase_plans()
    audits: list[dict[str, Any]] = []
    for plan in plans:
        layer_id = str(plan.get("plan_key", {}).get("layer_id", "unknown"))
        phase = str(plan.get("phase", "unknown"))
        plan_hash = str(plan.get("plan_hash", ""))
        relevant_events = [
            row
            for row in transport_results
            if str(row.get("layer_id", "")) == layer_id
            and str(row.get("phase", "")) == phase
            and str(row.get("plan_hash", "")) == plan_hash
        ]
        audits.append(
            build_execution_audit(
                ExecutionAuditInput(
                    execution_plan=plan,
                    transport_events=tuple(relevant_events),
                    phase_contract={
                        "phase": phase,
                        "layer_id": layer_id,
                        "policy_enabled": policy_enabled,
                    },
                )
            ).to_dict()
        )
    status = "passed"
    if any(item.get("status") == "failed" for item in audits):
        status = "failed"
    elif not audits:
        status = "not_applicable"
    return {"status": status, "audits": audits}


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
    write_json(run_dir / "run_manifest.json", _build_run_manifest(config, run_id=run_id, model_path=model_path))
    write_json(run_dir / "source_provenance.json", source_provenance(str(Path(__file__).resolve())))
    (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    if config.topology.launcher.kind != "torchrun":
        payload = NativeEPSummary(
            ep_size=int(config.topology.ep_size),
            dispatcher=str(config.runtime.dispatcher),
            backend="nccl",
            status="blocked_environment",
            reason="online_policy_correctness_requires_torchrun",
            details={"launcher": config.topology.launcher.kind},
        ).to_dict()
        write_json(run_dir / "summary.json", payload)
        return 2

    status = verify_env_main(["--model", model_path])
    if status != 0:
        payload = NativeEPSummary(
            ep_size=int(config.topology.ep_size),
            dispatcher=str(config.runtime.dispatcher),
            backend="nccl",
            status="blocked_environment",
            reason="verify_env_failed",
            details={"model": model_path},
        ).to_dict()
        write_json(run_dir / "summary.json", payload)
        return status

    rank = 0
    local_rank = 0
    world_size = 1
    runtime = None
    logits = None
    experiment_timeline: list[dict[str, Any]] = []

    def record_event(event: str, **detail: Any) -> None:
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

        stage_barrier("count_agreement", ok=world_size == config.topology.ep_size, detail=f"world_size={world_size} ep_size={config.topology.ep_size}")
        dtype = dtype_from_name(config.runtime.precision)
        if config.online_policy.p2.artifact:
            os.environ["ROUTERSENSE_P2_HINT_ARTIFACT"] = config.online_policy.p2.artifact
        prompts_load_start_ns = time.monotonic_ns()
        prompts = load_prompts(config.workload.prompts)
        prompts_load_end_ns = time.monotonic_ns()
        record_event("prompts_loaded", prompt_count=len(prompts), duration_us=(prompts_load_end_ns - prompts_load_start_ns) / 1000.0)

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
        position_ids = build_position_ids(tokens)
        attention_mask = None
        request_table_hash = hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()

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

        policy_name = resolve_effective_policy_name(config.online_policy.name, "disabled")
        policy_capabilities = _policy_capabilities(config)
        online_runtime_config = _build_online_runtime_config(config)
        runtime = attach_formal_online_runtime(
            model=model,
            runtime_config=online_runtime_config,
            rank=rank,
            local_rank=local_rank,
            run_id=run_id,
            model_revision=model_path,
            request_table_hash=request_table_hash,
            hostname=summarize_rank_environment(rank, local_rank)["host"],
            step_id="forward-0",
            microbatch_id="mb-0",
            observer=None,
        )

        partial_stop = False
        try:
            forward_start_ns = time.monotonic_ns()
            record_event("forward_start")
            runtime.begin_forward(forward_epoch=0)
            with torch.inference_mode():
                logits = model(tokens, position_ids, attention_mask)
            runtime.end_forward()
            forward_end_ns = time.monotonic_ns()
            record_event("forward_end", total_forward_us=(forward_end_ns - forward_start_ns) / 1000.0)
        except SelectedLayerStop:
            partial_stop = True
            logits = None
            runtime.end_forward()
            forward_end_ns = time.monotonic_ns()
            record_event("forward_partial_stop", total_forward_us=(forward_end_ns - forward_start_ns) / 1000.0)

        native_dispatch_summary = summarize_native_dispatchers(model, rank=rank)
        local_expert_ids = collect_local_expert_ids(model)
        adapter = getattr(runtime, "transport_adapter", None)
        transport_results = adapter.export_results() if adapter is not None else runtime.export_transport_execution_results()
        phase_context_rows = runtime.export_phase_contexts()
        phase_stats = phase_context_stats(phase_context_rows)
        audit_payload = _build_rank_audits(
            runtime=runtime,
            transport_results=transport_results,
            policy_enabled=bool(
                policy_name and online_runtime_config.execution_mode in {"phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"}
            ),
        )

        rank_summary = {
            **summarize_rank_environment(rank, local_rank),
            "ep_group_ranks": list(range(world_size)),
            "dispatcher_type": config.runtime.dispatcher,
            "local_expert_ids": local_expert_ids,
            "number_of_local_experts": len(local_expert_ids),
            "forward_completed": bool(logits is not None),
            "forward_partial_stop": partial_stop,
            "logits_status": "not_applicable" if partial_stop else ("produced" if isinstance(logits, torch.Tensor) else "missing"),
            "output_checksum": float(logits.float().sum().item()) if isinstance(logits, torch.Tensor) else None,
            "output_shape": list(logits.shape) if isinstance(logits, torch.Tensor) else None,
            "remote_dispatch_rows": phase_stats["p0_remote_rows"],
            "remote_combine_rows": phase_stats["p1_remote_rows"],
            "local_dispatch_rows": native_dispatch_summary["local_dispatch_rows"],
            "local_combine_rows": native_dispatch_summary["local_combine_rows"],
            **phase_stats,
            "policy_name": policy_name or "disabled",
            "policy_version": "v1" if policy_name else "",
            "policy_capabilities": policy_capabilities,
            "legacy_scheduler_mode": "",
            "execution_mode": online_runtime_config.execution_mode,
            "control_mode": online_runtime_config.control_mode,
            "bucket_rows": online_runtime_config.execution_selection.bucket_rows,
            "p0_weight": online_runtime_config.policy_parameters.p0_weight,
            "p1_reservation_weight": online_runtime_config.policy_parameters.p1_reservation_weight,
            "p2_hint_weight": online_runtime_config.policy_parameters.p2_hint_weight,
            "p2_hint_mode": online_runtime_config.policy_parameters.p2_hint_mode,
            "schedule_layer_selector": online_runtime_config.execution_selection.layer_selector,
            "schedule_phase_selector": online_runtime_config.execution_selection.phase_selector,
            "precision": config.runtime.precision,
            "transport_mutation": bool(
                policy_name and online_runtime_config.execution_mode in {"phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"}
            ),
            "dispatcher_rows": native_dispatch_summary["dispatcher_rows"],
            "transport_execution_count": len(transport_results),
            "execution_audit_status": audit_payload["status"],
        }
        rank_summary = write_rank_artifacts(
            run_dir=run_dir,
            run_id=run_id,
            rank=rank,
            logits=logits,
            runtime=runtime,
            native_dispatch_summary=native_dispatch_summary,
            rank_summary=rank_summary,
            save_logits=config.validation.save_logits,
            capture_layer_selector=config.observation.capture_layer_selector,
            capture_phase_selector=config.observation.capture_phase_selector,
        )
        write_json(run_dir / f"rank{rank}_execution_audit.json", audit_payload)
        write_json(run_dir / f"rank{rank}_experiment_timeline.json", {"events": experiment_timeline})
        gathered = gather_rank_payloads(rank_summary)

        if rank == 0:
            remote_dispatch_exercised = any(int(item.get("remote_dispatch_rows", 0)) > 0 for item in gathered)
            remote_combine_exercised = any(int(item.get("remote_combine_rows", 0)) > 0 for item in gathered)
            transport_mutation = bool(
                policy_name and online_runtime_config.execution_mode in {"phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"}
            )
            details = {
                "run_id": run_id,
                "model": model_path,
                "precision": config.runtime.precision,
                "world_size": world_size,
                "policy_name": policy_name or "disabled",
                "policy_version": "v1" if policy_name else "",
                "policy_capabilities": policy_capabilities,
                "legacy_scheduler_mode": "",
                "execution_mode": online_runtime_config.execution_mode,
                "control_mode": online_runtime_config.control_mode,
                "bucket_rows": online_runtime_config.execution_selection.bucket_rows,
                "p0_weight": online_runtime_config.policy_parameters.p0_weight,
                "p1_reservation_weight": online_runtime_config.policy_parameters.p1_reservation_weight,
                "p2_hint_weight": online_runtime_config.policy_parameters.p2_hint_weight,
                "p2_hint_mode": online_runtime_config.policy_parameters.p2_hint_mode,
                "schedule_layer_selector": online_runtime_config.execution_selection.layer_selector,
                "schedule_phase_selector": online_runtime_config.execution_selection.phase_selector,
                "transport_mutation": transport_mutation,
                "rank_summaries": gathered,
                "execution_audit_status": audit_payload["status"],
                "total_forward_us": next((float(item.get("total_forward_us")) for item in reversed(experiment_timeline) if "total_forward_us" in item), 0.0),
                "experiment_timeline_event_count": len(experiment_timeline),
            }
            payload = NativeEPSummary(
                ep_size=config.topology.ep_size,
                dispatcher=config.runtime.dispatcher,
                backend="nccl",
                forward_completed=all(bool(item.get("forward_completed") or item.get("forward_partial_stop")) for item in gathered),
                remote_dispatch_exercised=remote_dispatch_exercised,
                remote_combine_exercised=remote_combine_exercised,
                status="ready" if audit_payload["status"] != "failed" else "execution_audit_failed",
                reason=None if audit_payload["status"] != "failed" else "execution_audit_failed",
                details=details,
            ).to_dict()
            write_json(run_dir / "summary.json", payload)
            if audit_payload["status"] == "failed":
                write_json(run_dir / "failure_report.json", {"status": "execution_audit_failed", "details": audit_payload})
            else:
                write_not_triggered(run_dir / "failure_report.json")
            write_not_triggered(run_dir / "watchdog_report.json")
        stage_barrier("artifact_flush", ok=True, detail=str(run_dir))
        return 1 if audit_payload["status"] == "failed" else 0
    except Exception as exc:
        record_event("runtime_exception", exception=f"{type(exc).__name__}: {exc}")
        write_json(run_dir / f"rank{rank}_experiment_timeline.json", {"events": experiment_timeline})
        failure = failure_report(stage="phase_executor_runtime", exc=exc, rank=rank, local_rank=local_rank)
        write_json(run_dir / "failure_report.json", failure)
        if rank == 0:
            payload = NativeEPSummary(
                ep_size=config.topology.ep_size,
                dispatcher=config.runtime.dispatcher,
                backend="nccl",
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
