from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from rs.runtime.online.megatron_ep.observation import write_json, write_jsonl
from rs.scheduling.registry import supported_phase_policies


def local_expert_ids(model: torch.nn.Module) -> list[int]:
    found: set[int] = set()
    for module in model.modules():
        dispatcher = getattr(module, "token_dispatcher", None)
        if dispatcher is None:
            continue
        for idx in getattr(dispatcher, "local_expert_indices", []) or []:
            found.add(int(idx))
    return sorted(found)


def source_provenance(entrypoint: str) -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_available": bool(torch.distributed.is_nccl_available()),
        "entrypoint": entrypoint,
    }


def selector_matches(selector: str, value: str) -> bool:
    if selector in {"", "all", "both"}:
        return True
    selected = {item.strip() for item in selector.split(",") if item.strip()}
    return value in selected


def capture_enabled(*, layer_selector: str, phase_selector: str, layer_id: str, phase: str) -> bool:
    return selector_matches(layer_selector, layer_id) and selector_matches(phase_selector, phase)


def phase_context_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    cosine = torch.nn.functional.cosine_similarity(af.reshape(1, -1), bf.reshape(1, -1)).item()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine_similarity": float(cosine),
    }


def write_not_triggered(path: Path) -> None:
    write_json(path, {"status": "not_triggered"})


def effective_policy_name(policy: str, scheduler_mode: str) -> str:
    if policy:
        return str(policy)
    if scheduler_mode in set(supported_phase_policies()):
        return str(scheduler_mode)
    return ""


def failure_report(
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


def write_rank_artifacts(
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
        prepared_plan_summary = runtime.export_prepared_plan_summary()
        if prepared_plan_summary:
            rank_summary.update(prepared_plan_summary)
        write_jsonl(run_dir / f"rank{rank}_control_timeline.jsonl", runtime.export_control_timeline())
        write_jsonl(run_dir / f"rank{rank}_control_commands.jsonl", runtime.export_control_commands())
        write_json(run_dir / f"rank{rank}_assertions.json", runtime.export_assertions())
        write_jsonl(run_dir / f"rank{rank}_phase_contexts.jsonl", runtime.export_phase_contexts())
        write_jsonl(run_dir / f"rank{rank}_transport_bundles.jsonl", runtime.export_transport_bundles())
        write_jsonl(run_dir / f"rank{rank}_scheduled_phase_plans.jsonl", runtime.export_scheduled_phase_plans())
        write_jsonl(run_dir / f"rank{rank}_plan_arrival_records.jsonl", runtime.export_plan_arrival_records())
        write_jsonl(run_dir / f"rank{rank}_window_state.jsonl", runtime.export_window_state_records())
        write_jsonl(run_dir / f"rank{rank}_prepared_plan_bindings.jsonl", runtime.export_prepared_plan_bindings())
        write_jsonl(run_dir / f"rank{rank}_release_events.jsonl", runtime.export_release_events())
        write_jsonl(run_dir / f"rank{rank}_window_schedule_shadow.jsonl", runtime.export_window_schedule_shadows())
        write_jsonl(run_dir / f"rank{rank}_prepared_phase_plan_shadow.jsonl", runtime.export_prepared_phase_plan_shadows())
        write_jsonl(run_dir / f"rank{rank}_pending_window_driver.jsonl", runtime.export_pending_window_driver_records())
        write_jsonl(run_dir / f"rank{rank}_planning_timing.jsonl", runtime.export_planning_timing_records())
        replay_trace_rows = runtime.export_control_replay_traces()
        if replay_trace_rows:
            write_jsonl(run_dir / f"rank{rank}_control_replay_trace.jsonl", replay_trace_rows)
        prediction_audit_rows = runtime.export_prediction_audits()
        if prediction_audit_rows:
            write_jsonl(run_dir / f"rank{rank}_prediction_audit.jsonl", prediction_audit_rows)
        write_json(run_dir / f"rank{rank}_prepared_plan_summary.json", prepared_plan_summary)
        adapter = getattr(runtime, "transport_adapter", None)
        transport_results = adapter.export_results() if adapter is not None else runtime.export_transport_execution_results()
        write_jsonl(run_dir / f"rank{rank}_transport_execution.jsonl", transport_results)
        write_jsonl(run_dir / f"rank{rank}_captured_phase_tensors.jsonl", runtime.export_captured_phase_tensors())
        capture_dir = run_dir / "captured_phase_tensors"
        capture_dir.mkdir(parents=True, exist_ok=True)
        for item in runtime.export_captured_phase_tensors_with_payload():
            layer_id = str(item["layer_id"])
            phase = str(item["phase"])
            if not capture_enabled(
                layer_selector=capture_layer_selector,
                phase_selector=capture_phase_selector,
                layer_id=layer_id,
                phase=phase,
            ):
                continue
            tensor_path = capture_dir / f"rank{rank}_layer{layer_id}_{phase}_{item['tensor_role']}.pt"
            torch.save(item["tensor"], tensor_path)
    return rank_summary


def checksum_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()
