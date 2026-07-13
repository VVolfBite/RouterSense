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
from rs.runtime.online.megatron_ep.observation.tokenization import compute_token_count_contract
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
            selected_layer_ids=tuple(str(item) for item in config.execution.schedule.selected_layer_ids),
            bucket_mode=str(config.execution.bucket_mode),
            bucket_rows=config.execution.bucket_rows,
        ),
        policy_parameters=OnlinePolicyParameters(
            p0_weight=config.online_policy.parameters.p0_weight,
            p1_reservation_weight=config.online_policy.parameters.p1_reservation_weight,
            p2_hint_weight=config.online_policy.parameters.p2_hint_weight,
            residual_weight=config.online_policy.parameters.residual_weight,
            barrier_weight=config.online_policy.parameters.barrier_weight,
            age_weight=config.online_policy.parameters.age_weight,
            prediction_weight=config.online_policy.parameters.prediction_weight,
            p2_hint_mode=config.online_policy.p2.mode,
            p2_hint_artifact=config.online_policy.p2.artifact,
            calibrated_p2_enabled=config.online_policy.p2.mode == "calibrated_artifact",
            online_p2_predictor=config.online_policy.parameters.online_p2_predictor,
            safe_projection_mode=str(config.execution.safe_projection_mode),
        ),
        observation=config.observation.to_dict(),
        validation=OnlineValidationConfig(
            stop_after_selected_layer=bool(config.validation.stop_after_selected_layer),
            executor_heartbeat_path=str(config.artifact.output_root),
            executor_phase_timeout_sec=120,
            preflight_mode=str(config.execution.preflight_mode),
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
        "runtime_line": str(config.runtime.line),
        "invariant_mode": str(config.runtime.invariant_mode),
        "policy_name": config.online_policy.name,
        "execution_mode": config.execution.mode,
        "bucket_mode": str(config.execution.bucket_mode),
        "control_mode": config.runtime.control_mode,
        "safe_projection_mode": str(config.execution.safe_projection_mode),
        "observation_profile": config.observation.profile,
        "model_id": config.model.model_id,
        "model_path": model_path,
        "precision": config.runtime.precision,
        "ep_size": config.topology.ep_size,
        "prompt_file": config.workload.prompts,
        "bucket_rows": config.execution.bucket_rows,
        "schedule_layer_selector": config.execution.schedule.layer_selector,
        "selected_layer_ids": list(config.execution.schedule.selected_layer_ids),
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

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except Exception:
            return int(default)

    def _snapshot_perf(runtime_obj: Any | None) -> dict[str, float]:
        if runtime_obj is None:
            return {}
        counters = getattr(runtime_obj, "perf_counters", {}) or {}
        return {str(stage): float((payload or {}).get("total_us", 0.0) or 0.0) for stage, payload in counters.items()}

    def _delta_perf(after: dict[str, float], before: dict[str, float], stage: str) -> float:
        return max(0.0, float(after.get(stage, 0.0) - before.get(stage, 0.0)))

    def _snapshot_adapter(adapter_obj: Any | None) -> dict[str, int]:
        if adapter_obj is None:
            return {}
        return {
            "async_executor_invocation_count": int(getattr(adapter_obj, "async_executor_invocation_count", 0)),
            "batch_isend_irecv_call_count": int(getattr(adapter_obj, "batch_isend_irecv_call_count", 0)),
            "real_send_op_count": int(getattr(adapter_obj, "real_send_op_count", 0)),
            "real_recv_op_count": int(getattr(adapter_obj, "real_recv_op_count", 0)),
            "local_copy_task_count": int(getattr(adapter_obj, "local_copy_task_count", 0)),
            "phase_sync_fallback_count": int(getattr(adapter_obj, "phase_sync_fallback_count", 0)),
        }

    def _delta_adapter(after: dict[str, int], before: dict[str, int], key: str) -> int:
        return max(0, int(after.get(key, 0)) - int(before.get(key, 0)))

    def _snapshot_transport_index(runtime_obj: Any | None) -> int:
        if runtime_obj is None:
            return 0
        adapter_obj = getattr(runtime_obj, "transport_adapter", None)
        if adapter_obj is None:
            return 0
        return len(adapter_obj.export_results())

    def _aggregate_transport_results(runtime_obj: Any | None, start_index: int) -> dict[str, float | int | bool]:
        if runtime_obj is None:
            return {}
        adapter_obj = getattr(runtime_obj, "transport_adapter", None)
        if adapter_obj is None:
            return {}
        rows = list(adapter_obj.export_results())[int(start_index):]
        result_rows = [row for row in rows if str(row.get("record_type", "")) == "result_summary"]
        if not result_rows:
            return {}
        dispatch_transport_us = 0.0
        return_transport_us = 0.0
        local_copy_us = 0.0
        op_build_us = 0.0
        batch_submit_us = 0.0
        wait_us = 0.0
        wave_concat_us = 0.0
        wave_collective_us = 0.0
        wave_scatter_us = 0.0
        idle_barrier_wait_us = 0.0
        task_count = 0
        p2p_op_count = 0
        wave_count = 0
        collective_count = 0
        batch_isend_irecv_count = 0
        preflight_collective_count = 0
        all_work_completed = True
        timeout_observed = False
        first_transport_submit_ns = 0
        last_transport_complete_ns = 0
        p0_first_submit_ns = 0
        p0_last_complete_ns = 0
        p1_first_submit_ns = 0
        p1_last_complete_ns = 0
        for row in result_rows:
            phase = str(row.get("phase", "")).upper()
            timing_us = dict(row.get("timing_us", {}) or {})
            phase_metrics = dict(row.get("phase_metrics", {}) or {})
            communication_total = float(timing_us.get("communication_us", timing_us.get("total_forward_us", 0.0)) or 0.0)
            if phase == "P0":
                dispatch_transport_us += communication_total
            elif phase == "P1":
                return_transport_us += communication_total
            local_copy_us += float(timing_us.get("local_copy_us", 0.0) or 0.0)
            op_build_us += float(timing_us.get("op_build_us", 0.0) or 0.0)
            batch_submit_us += float(timing_us.get("batch_submit_us", timing_us.get("submit_us", 0.0)) or 0.0)
            wait_us += float(timing_us.get("work_wait_us", timing_us.get("wait_us", 0.0)) or 0.0)
            wave_concat_us += float(timing_us.get("wave_concat_us", 0.0) or 0.0)
            wave_collective_us += float(timing_us.get("wave_collective_us", 0.0) or 0.0)
            wave_scatter_us += float(timing_us.get("wave_scatter_us", 0.0) or 0.0)
            idle_barrier_wait_us += float(timing_us.get("idle_barrier_wait_us", 0.0) or 0.0)
            task_count += int(phase_metrics.get("task_count", 0) or 0)
            p2p_op_count += int(phase_metrics.get("p2p_op_count", 0) or 0)
            wave_count += int(phase_metrics.get("wave_count", 0) or 0)
            collective_count += int(phase_metrics.get("collective_count", 0) or 0)
            batch_isend_irecv_count += int(row.get("batch_isend_irecv_call_count", 0) or 0)
            preflight_collective_count += int(row.get("preflight_collective_count", 0) or 0)
            all_work_completed = all_work_completed and bool(row.get("all_work_completed", True))
            timeout_observed = timeout_observed or bool(row.get("timeout", False))
            submit_ns = int(row.get("first_transport_submit_ns", 0) or 0)
            complete_ns = int(row.get("last_transport_complete_ns", 0) or 0)
            if submit_ns > 0:
                first_transport_submit_ns = submit_ns if first_transport_submit_ns <= 0 else min(first_transport_submit_ns, submit_ns)
            if complete_ns > 0:
                last_transport_complete_ns = max(last_transport_complete_ns, complete_ns)
            p0_submit_ns = int(row.get("p0_first_submit_ns", 0) or 0)
            p0_complete_ns = int(row.get("p0_last_complete_ns", 0) or 0)
            p1_submit_ns = int(row.get("p1_first_submit_ns", 0) or 0)
            p1_complete_ns = int(row.get("p1_last_complete_ns", 0) or 0)
            if p0_submit_ns > 0:
                p0_first_submit_ns = p0_submit_ns if p0_first_submit_ns <= 0 else min(p0_first_submit_ns, p0_submit_ns)
            if p0_complete_ns > 0:
                p0_last_complete_ns = max(p0_last_complete_ns, p0_complete_ns)
            if p1_submit_ns > 0:
                p1_first_submit_ns = p1_submit_ns if p1_first_submit_ns <= 0 else min(p1_first_submit_ns, p1_submit_ns)
            if p1_complete_ns > 0:
                p1_last_complete_ns = max(p1_last_complete_ns, p1_complete_ns)
        return {
            "dispatch_transport_us": dispatch_transport_us,
            "return_transport_us": return_transport_us,
            "local_copy_us": local_copy_us,
            "op_build_us": op_build_us,
            "batch_submit_us": batch_submit_us,
            "wait_us": wait_us,
            "wave_concat_us": wave_concat_us,
            "wave_collective_us": wave_collective_us,
            "wave_scatter_us": wave_scatter_us,
            "idle_barrier_wait_us": idle_barrier_wait_us,
            "task_count": task_count,
            "p2p_op_count": p2p_op_count,
            "wave_count": wave_count,
            "collective_count": collective_count,
            "batch_isend_irecv_count": batch_isend_irecv_count,
            "preflight_collective_count": preflight_collective_count,
            "all_work_completed": all_work_completed,
            "timeout_observed": timeout_observed,
            "first_transport_submit_ns": int(first_transport_submit_ns),
            "last_transport_complete_ns": int(last_transport_complete_ns),
            "p0_first_submit_ns": int(p0_first_submit_ns),
            "p0_last_complete_ns": int(p0_last_complete_ns),
            "p1_first_submit_ns": int(p1_first_submit_ns),
            "p1_last_complete_ns": int(p1_last_complete_ns),
        }
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
        tokenization = config.workload.tokenization
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": tokenization.padding if tokenization.padding else True,
            "truncation": bool(tokenization.truncation),
        }
        if tokenization.max_length is not None:
            tokenizer_kwargs["max_length"] = int(tokenization.max_length)
        encoded = tokenizer(prompts, **tokenizer_kwargs)
        tokenizer_end_ns = time.monotonic_ns()
        actual_prompt_count = int(len(prompts))
        actual_batch_rows = int(encoded["input_ids"].shape[0])
        actual_seq_len = int(encoded["input_ids"].shape[1])
        attention_mask_sum = int(encoded.get("attention_mask").sum().item()) if encoded.get("attention_mask") is not None else None
        token_count_contract = compute_token_count_contract(
            actual_batch_rows=actual_batch_rows,
            actual_seq_len=actual_seq_len,
            attention_mask_sum=attention_mask_sum,
        )
        tokenization_shape_valid = (
            (tokenization.expected_prompt_count is None or actual_prompt_count == int(tokenization.expected_prompt_count))
            and (tokenization.expected_batch_rows is None or actual_batch_rows == int(tokenization.expected_batch_rows))
            and (tokenization.expected_seq_len is None or actual_seq_len == int(tokenization.expected_seq_len))
        )
        record_event(
            "tokenizer_and_encode_done",
            duration_us=(tokenizer_end_ns - tokenizer_start_ns) / 1000.0,
            batch_rows=actual_batch_rows,
            seq_len=actual_seq_len,
            tokenization_shape_valid=bool(tokenization_shape_valid),
        )
        tokens = encoded["input_ids"].to(device=f"cuda:{local_rank}")
        position_ids = build_position_ids(tokens)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=f"cuda:{local_rank}")
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

        warmup_iters = max(0, _env_int("ROUTERSENSE_WARMUP_ITERS", 0))
        measure_iters = max(1, _env_int("ROUTERSENSE_MEASURE_ITERS", 1))
        total_iters = warmup_iters + measure_iters
        repeat_records: list[dict[str, Any]] = []
        partial_stop = False
        logits = None
        final_output_checksum: float | None = None
        final_output_shape: list[int] | None = None
        final_logits_available = False
        for iter_index in range(total_iters):
            is_warmup = iter_index < warmup_iters
            measure_index = -1 if is_warmup else (iter_index - warmup_iters)
            forward_epoch = int(iter_index)
            runtime.step_id = f"forward-{forward_epoch}"
            runtime.microbatch_id = f"mb-{forward_epoch}"
            local_forward_us = 0.0
            global_forward_us = 0.0
            try:
                adapter_before = _snapshot_adapter(getattr(runtime, "transport_adapter", None))
                transport_before_index = _snapshot_transport_index(runtime)
                perf_before = _snapshot_perf(runtime)
                record_event(
                    "forward_start",
                    forward_epoch=forward_epoch,
                    warmup=is_warmup,
                    measure_index=measure_index,
                )
                runtime.begin_forward(forward_epoch=forward_epoch)
                host_start_ns = time.monotonic_ns()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(local_rank)
                start_event.record()
                with torch.inference_mode():
                    logits = model(tokens, position_ids, attention_mask)
                end_event.record()
                end_event.synchronize()
                host_end_ns = time.monotonic_ns()
                local_forward_us = float(start_event.elapsed_time(end_event) * 1000.0)
                local_host_forward_us = float((host_end_ns - host_start_ns) / 1000.0)
                duration_tensor = torch.tensor([local_forward_us], device=tokens.device, dtype=torch.float64)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(duration_tensor, op=torch.distributed.ReduceOp.MAX)
                global_forward_us = float(duration_tensor.item())
                runtime.end_forward()
                adapter_after = _snapshot_adapter(getattr(runtime, "transport_adapter", None))
                transport_timing = _aggregate_transport_results(runtime, transport_before_index)
                perf_after = _snapshot_perf(runtime)
                output_shape = list(logits.shape) if isinstance(logits, torch.Tensor) else None
                output_checksum = float(logits.float().sum().item()) if isinstance(logits, torch.Tensor) else None
                final_output_shape = output_shape
                final_output_checksum = output_checksum
                final_logits_available = isinstance(logits, torch.Tensor)
                repeat_record = {
                    "forward_epoch": forward_epoch,
                    "warmup": bool(is_warmup),
                    "measure_index": int(measure_index),
                    "local_forward_us": float(local_forward_us),
                    "local_host_forward_us": float(local_host_forward_us),
                    "global_max_forward_us": float(global_forward_us),
                    "traffic_observation_us": _delta_perf(perf_after, perf_before, "build_runtime_observation"),
                    "matrix_gather_us": _delta_perf(perf_after, perf_before, "after_p0_observation"),
                    "gpu_to_cpu_us": 0.0,
                    "prediction_us": _delta_perf(perf_after, perf_before, "predict_next_dispatch"),
                    "raw_u_build_us": _delta_perf(perf_after, perf_before, "raw_u_build"),
                    "paired_b_build_us": _delta_perf(perf_after, perf_before, "paired_b_build"),
                    "safe_projection_us": _delta_perf(perf_after, perf_before, "host_projection"),
                    "host_projection_us": _delta_perf(perf_after, perf_before, "host_projection"),
                    "safe_selection_us": _delta_perf(perf_after, perf_before, "safe_selection"),
                    "compiler_us": _delta_perf(perf_after, perf_before, "activate_transport"),
                    "plan_agreement_us": _delta_perf(perf_after, perf_before, "agree_global_joint_plan_digest"),
                    "local_materialization_us": _delta_perf(perf_after, perf_before, "activate_transport"),
                    "dispatch_hook_path_us": _delta_perf(perf_after, perf_before, "hook_before_token_dispatch_total"),
                    "combine_hook_path_us": _delta_perf(perf_after, perf_before, "hook_before_token_combine_total"),
                    "preflight_us": _delta_perf(perf_after, perf_before, "activate_transport"),
                    "dispatch_transport_us": float(transport_timing.get("dispatch_transport_us", 0.0) or 0.0),
                    "return_transport_us": float(transport_timing.get("return_transport_us", 0.0) or 0.0),
                    "all_rank_transport_span_us": float((transport_timing.get("dispatch_transport_us", 0.0) or 0.0) + (transport_timing.get("return_transport_us", 0.0) or 0.0)),
                    "local_copy_us": float(transport_timing.get("local_copy_us", 0.0) or 0.0),
                    "op_build_us": float(transport_timing.get("op_build_us", 0.0) or 0.0),
                    "submit_us": float(transport_timing.get("batch_submit_us", 0.0) or 0.0),
                    "batch_submit_us": float(transport_timing.get("batch_submit_us", 0.0) or 0.0),
                    "wait_us": float(transport_timing.get("wait_us", 0.0) or 0.0),
                    "expert_compute_us": max(
                        0.0,
                        float(local_forward_us)
                        - (
                            float(_delta_perf(perf_after, perf_before, "hook_before_token_dispatch_total"))
                            + float(_delta_perf(perf_after, perf_before, "hook_before_token_combine_total"))
                        ),
                    ),
                    "combine_us": float(_delta_perf(perf_after, perf_before, "hook_before_token_combine_total")),
                    "artifact_hot_path_us": 0.0,
                    "wave_concat_us": float(transport_timing.get("wave_concat_us", 0.0) or 0.0),
                    "wave_collective_us": float(transport_timing.get("wave_collective_us", 0.0) or 0.0),
                    "wave_scatter_us": float(transport_timing.get("wave_scatter_us", 0.0) or 0.0),
                    "idle_barrier_wait_us": float(transport_timing.get("idle_barrier_wait_us", 0.0) or 0.0),
                    "task_count": int(transport_timing.get("task_count", 0) or 0),
                    "p2p_op_count": int(transport_timing.get("p2p_op_count", 0) or 0),
                    "wave_count": int(transport_timing.get("wave_count", 0) or 0),
                    "collective_count": int(transport_timing.get("collective_count", 0) or 0),
                    "batch_isend_irecv_count": int(transport_timing.get("batch_isend_irecv_count", 0) or 0),
                    "preflight_collective_count": int(transport_timing.get("preflight_collective_count", 0) or 0),
                    "all_work_completed": bool(transport_timing.get("all_work_completed", True)),
                    "timeout_observed": bool(transport_timing.get("timeout_observed", False)),
                    "first_transport_submit_ns": int(transport_timing.get("first_transport_submit_ns", 0) or 0),
                    "last_transport_complete_ns": int(transport_timing.get("last_transport_complete_ns", 0) or 0),
                    "p0_first_submit_ns": int(transport_timing.get("p0_first_submit_ns", 0) or 0),
                    "p0_last_complete_ns": int(transport_timing.get("p0_last_complete_ns", 0) or 0),
                    "p1_first_submit_ns": int(transport_timing.get("p1_first_submit_ns", 0) or 0),
                    "p1_last_complete_ns": int(transport_timing.get("p1_last_complete_ns", 0) or 0),
                    "unattributed_us": max(
                        0.0,
                        float(global_forward_us)
                        - (
                            float(_delta_perf(perf_after, perf_before, "build_runtime_observation"))
                            + float(_delta_perf(perf_after, perf_before, "after_p0_observation"))
                            + float(_delta_perf(perf_after, perf_before, "predict_next_dispatch"))
                            + float(_delta_perf(perf_after, perf_before, "raw_u_build"))
                            + float(_delta_perf(perf_after, perf_before, "paired_b_build"))
                            + float(_delta_perf(perf_after, perf_before, "host_projection"))
                            + float(_delta_perf(perf_after, perf_before, "safe_selection"))
                            + float(_delta_perf(perf_after, perf_before, "activate_transport"))
                            + float(_delta_perf(perf_after, perf_before, "agree_global_joint_plan_digest"))
                            + float(transport_timing.get("dispatch_transport_us", 0.0) or 0.0)
                            + float(transport_timing.get("return_transport_us", 0.0) or 0.0)
                        ),
                    ),
                    "async_executor_invocation_count": _delta_adapter(adapter_after, adapter_before, "async_executor_invocation_count"),
                    "batch_isend_irecv_call_count": _delta_adapter(adapter_after, adapter_before, "batch_isend_irecv_call_count"),
                    "real_send_op_count": _delta_adapter(adapter_after, adapter_before, "real_send_op_count"),
                    "real_recv_op_count": _delta_adapter(adapter_after, adapter_before, "real_recv_op_count"),
                    "local_copy_task_count": _delta_adapter(adapter_after, adapter_before, "local_copy_task_count"),
                    "phase_sync_fallback_count": _delta_adapter(adapter_after, adapter_before, "phase_sync_fallback_count"),
                    "logits_shape": output_shape,
                    "output_checksum": output_checksum,
                }
                if config.validation.save_logits and isinstance(logits, torch.Tensor) and not is_warmup:
                    repeat_logits_path = run_dir / f"{run_id}-rank{rank}-epoch{forward_epoch}-measure{measure_index}.pt"
                    torch.save(logits.detach().float().cpu(), repeat_logits_path)
                    repeat_record["logits_path"] = str(repeat_logits_path)
                repeat_records.append(repeat_record)
                record_event(
                    "forward_end",
                    forward_epoch=forward_epoch,
                    warmup=is_warmup,
                    measure_index=measure_index,
                    total_forward_us=float(global_forward_us),
                    local_forward_us=float(local_forward_us),
                )
                if not config.validation.save_logits:
                    logits = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except SelectedLayerStop:
                partial_stop = True
                logits = None
                runtime.end_forward()
                repeat_records.append(
                    {
                        "forward_epoch": forward_epoch,
                        "warmup": bool(is_warmup),
                        "measure_index": int(measure_index),
                        "partial_stop": True,
                    }
                )
                record_event(
                    "forward_partial_stop",
                    forward_epoch=forward_epoch,
                    warmup=is_warmup,
                    measure_index=measure_index,
                    total_forward_us=float(global_forward_us),
                    local_forward_us=float(local_forward_us),
                )
                break

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
            "forward_completed": bool(final_logits_available),
            "forward_partial_stop": partial_stop,
            "logits_status": "not_applicable" if partial_stop else ("produced" if final_logits_available else "missing"),
            "output_checksum": final_output_checksum,
            "output_shape": final_output_shape,
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
            "bucket_mode": str(getattr(online_runtime_config.execution_selection, "bucket_mode", "dynamic_current")),
            "bucket_rows": online_runtime_config.execution_selection.bucket_rows,
            "safe_projection_mode": str(getattr(online_runtime_config.policy_parameters, "safe_projection_mode", "host_select")),
            "p0_weight": online_runtime_config.policy_parameters.p0_weight,
            "p1_reservation_weight": online_runtime_config.policy_parameters.p1_reservation_weight,
            "p2_hint_weight": online_runtime_config.policy_parameters.p2_hint_weight,
            "residual_weight": float(getattr(online_runtime_config.policy_parameters, "residual_weight", 0.75)),
            "barrier_weight": float(getattr(online_runtime_config.policy_parameters, "barrier_weight", 1.75)),
            "age_weight": float(getattr(online_runtime_config.policy_parameters, "age_weight", 0.15)),
            "prediction_weight": float(getattr(online_runtime_config.policy_parameters, "prediction_weight", 0.35)),
            "p2_hint_mode": online_runtime_config.policy_parameters.p2_hint_mode,
            "schedule_layer_selector": online_runtime_config.execution_selection.layer_selector,
            "selected_layer_ids": list(online_runtime_config.execution_selection.selected_layer_ids),
            "schedule_phase_selector": online_runtime_config.execution_selection.phase_selector,
            "selected_layer_match_count": int(getattr(runtime._runtime_state.metrics, "selected_layer_match_count", 0) if runtime is not None else 0),
            "selected_p0_hook_count": int(getattr(runtime._runtime_state.metrics, "selected_p0_hook_count", 0) if runtime is not None else 0),
            "selected_p1_hook_count": int(getattr(runtime._runtime_state.metrics, "selected_p1_hook_count", 0) if runtime is not None else 0),
            "selected_transport_execution_count": int(getattr(runtime._runtime_state.metrics, "selected_transport_execution_count", 0) if runtime is not None else 0),
            "precision": config.runtime.precision,
            "runtime_line": str(config.runtime.line),
            "invariant_mode": str(config.runtime.invariant_mode),
            "transport_mutation": bool(
                policy_name and online_runtime_config.execution_mode in {"phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"}
            ),
            "dispatcher_rows": native_dispatch_summary["dispatcher_rows"],
            "transport_execution_count": len(transport_results),
            "execution_audit_status": audit_payload["status"],
            "repeat_records": repeat_records,
            "requested_prompt_count": tokenization.expected_prompt_count,
            "actual_prompt_count": actual_prompt_count,
            "actual_batch_rows": actual_batch_rows,
            "actual_seq_len": actual_seq_len,
            "tokenization_shape_valid": bool(tokenization_shape_valid),
            "tokenization_padding_mode": str(tokenization.padding),
            "tokenization_truncation": bool(tokenization.truncation),
            "tokenization_max_length": tokenization.max_length,
            **token_count_contract.to_dict(),
        }
        rank_summary = write_rank_artifacts(
            run_dir=run_dir,
            run_id=run_id,
            rank=rank,
            logits=logits if config.validation.save_logits else None,
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
                "bucket_mode": str(getattr(online_runtime_config.execution_selection, "bucket_mode", "dynamic_current")),
                "bucket_rows": online_runtime_config.execution_selection.bucket_rows,
                "safe_projection_mode": str(getattr(online_runtime_config.policy_parameters, "safe_projection_mode", "host_select")),
                "p0_weight": online_runtime_config.policy_parameters.p0_weight,
                "p1_reservation_weight": online_runtime_config.policy_parameters.p1_reservation_weight,
                "p2_hint_weight": online_runtime_config.policy_parameters.p2_hint_weight,
                "residual_weight": float(getattr(online_runtime_config.policy_parameters, "residual_weight", 0.75)),
                "barrier_weight": float(getattr(online_runtime_config.policy_parameters, "barrier_weight", 1.75)),
                "age_weight": float(getattr(online_runtime_config.policy_parameters, "age_weight", 0.15)),
                "prediction_weight": float(getattr(online_runtime_config.policy_parameters, "prediction_weight", 0.35)),
                "p2_hint_mode": online_runtime_config.policy_parameters.p2_hint_mode,
                "schedule_layer_selector": online_runtime_config.execution_selection.layer_selector,
                "selected_layer_ids": list(online_runtime_config.execution_selection.selected_layer_ids),
                "schedule_phase_selector": online_runtime_config.execution_selection.phase_selector,
                "selected_layer_match_count": int(getattr(runtime._runtime_state.metrics, "selected_layer_match_count", 0) if runtime is not None else 0),
                "selected_p0_hook_count": int(getattr(runtime._runtime_state.metrics, "selected_p0_hook_count", 0) if runtime is not None else 0),
                "selected_p1_hook_count": int(getattr(runtime._runtime_state.metrics, "selected_p1_hook_count", 0) if runtime is not None else 0),
                "selected_transport_execution_count": int(getattr(runtime._runtime_state.metrics, "selected_transport_execution_count", 0) if runtime is not None else 0),
                "requested_prompt_count": tokenization.expected_prompt_count,
                "actual_prompt_count": actual_prompt_count,
                "actual_batch_rows": actual_batch_rows,
                "actual_seq_len": actual_seq_len,
                "tokenization_shape_valid": bool(tokenization_shape_valid),
                "tokenization_padding_mode": str(tokenization.padding),
                "tokenization_truncation": bool(tokenization.truncation),
                "tokenization_max_length": tokenization.max_length,
                **token_count_contract.to_dict(),
                "runtime_line": str(config.runtime.line),
                "invariant_mode": str(config.runtime.invariant_mode),
                "transport_mutation": transport_mutation,
                "rank_summaries": gathered,
                "execution_audit_status": audit_payload["status"],
                "repeat_records": repeat_records,
                "warmup_iters": int(warmup_iters),
                "measure_iters": int(measure_iters),
                "total_forward_us": (
                    float(sorted(item["global_max_forward_us"] for item in repeat_records if not bool(item.get("warmup", False)) and "global_max_forward_us" in item)[
                        len([item for item in repeat_records if not bool(item.get("warmup", False)) and "global_max_forward_us" in item]) // 2
                    ])
                    if any((not bool(item.get("warmup", False)) and "global_max_forward_us" in item) for item in repeat_records)
                    else next((float(item.get("total_forward_us")) for item in reversed(experiment_timeline) if "total_forward_us" in item), 0.0)
                ),
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
        if runtime is not None:
            try:
                failure["prepared_plan_summary"] = runtime.export_prepared_plan_summary()
            except Exception:
                pass
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
