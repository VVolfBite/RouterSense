from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable

from rs.scheduling.contracts import (
    FlowDemand,
    FlowWindow,
    LogicalSchedulePlan,
    LogicalWave,
    MultiPhaseSchedulingProblem,
)
from rs.scheduling.diagnostics import PolicyDiagnostics, WaveDiagnostics
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE
from rs.scheduling.multiphase.replay import replay_and_audit_schedule
from rs.scheduling.phase_execution import BucketTask, PackedTensorDescriptor, PayloadSlice, PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.phase_execution_utils import validate_phase_execution_plan
from rs.scheduling.validation import stable_hash

from ..capabilities import PolicyCapabilities


def fifo_task_key(task: BucketTask) -> tuple[int, int, int, int]:
    return (int(task.src_rank), int(task.dst_rank), int(task.segment_ordinal), int(task.bucket_ordinal))


def reverse_bucket_task_key(task: BucketTask) -> tuple[int, int, int, int]:
    return (int(task.src_rank), int(task.dst_rank), -int(task.segment_ordinal), -int(task.bucket_ordinal))


def build_transfer_layouts_and_tasks(
    *,
    local_context: PhaseReadyContext,
    global_contexts: tuple[PhaseReadyContext, ...],
    bucket_rows: int,
) -> tuple[tuple[Any, ...], list[BucketTask]]:
    def _payload_specs(context: PhaseReadyContext) -> tuple[PackedTensorDescriptor, ...]:
        if context.payload_specs:
            return tuple(context.payload_specs)
        if context.transport_bundles:
            return tuple(context.transport_bundles[0].payloads)
        if local_context.payload_specs:
            return tuple(local_context.payload_specs)
        return ()

    def _rows_to_bytes(row_count: int, *, shape_suffix: tuple[int, ...], element_size_bytes: int) -> int:
        multiplier = 1
        for dim in shape_suffix:
            multiplier *= int(dim)
        return int(row_count) * int(multiplier) * int(element_size_bytes)

    transfer_layouts: list[Any] = []
    all_tasks: list[BucketTask] = []
    for context in global_contexts:
        payload_specs = _payload_specs(context)
        for segment in context.outgoing_segments:
            if str(segment.phase) != str(local_context.phase) or int(segment.src_rank) == int(segment.dst_rank) or int(segment.row_count) <= 0:
                continue
            transfer_layouts.append(
                SimpleNamespace(
                    transfer_key=f"{segment.phase}:{segment.src_rank}->{segment.dst_rank}",
                    bundle_id=f"{segment.segment_id}:bundle",
                    phase=segment.phase,
                    src_rank=int(segment.src_rank),
                    dst_rank=int(segment.dst_rank),
                    segment_ordinal=int(segment.segment_ordinal),
                    sender_offset_rows=int(segment.send_offset_rows),
                    receiver_offset_rows=0,
                    row_count=int(segment.row_count),
                    byte_count=int(segment.byte_count),
                    packed_send_layout_id=str(segment.packed_send_layout_id),
                    canonical_receive_layout_id="",
                    atomic_submit=bool(getattr(context, "atomic_submit", True)),
                    payloads=payload_specs,
                )
            )
            step = int(segment.row_count) if int(bucket_rows) <= 0 else int(bucket_rows)
            consumed = 0
            bucket_ordinal = 0
            while consumed < int(segment.row_count):
                current_rows = min(step, int(segment.row_count) - consumed)
                sender_offset = int(segment.send_offset_rows) + consumed
                payload_slices = tuple(
                    PayloadSlice(
                        bundle_id=f"{segment.segment_id}:bundle",
                        tensor_role=str(payload.tensor_role),
                        src_rank=int(segment.src_rank),
                        dst_rank=int(segment.dst_rank),
                        segment_ordinal=int(segment.segment_ordinal),
                        sender_offset_rows=sender_offset,
                        receiver_offset_rows=0,
                        row_count=int(current_rows),
                        dtype=str(payload.dtype),
                        shape_suffix=tuple(int(v) for v in payload.shape_suffix),
                        element_size_bytes=int(payload.element_size_bytes),
                        payload_byte_count=_rows_to_bytes(
                            int(current_rows),
                            shape_suffix=tuple(int(v) for v in payload.shape_suffix),
                            element_size_bytes=int(payload.element_size_bytes),
                        ),
                        packed_layout_id=str(segment.packed_send_layout_id),
                    )
                    for payload in payload_specs
                )
                all_tasks.append(
                    BucketTask(
                        task_id=f"{segment.phase}:{segment.src_rank}->{segment.dst_rank}:bucket:{bucket_ordinal}",
                        bundle_id=f"{segment.segment_id}:bundle",
                        phase=segment.phase,
                        src_rank=int(segment.src_rank),
                        dst_rank=int(segment.dst_rank),
                        source_peer_index=-1,
                        destination_peer_index=int(segment.destination_peer_index),
                        segment_ordinal=int(segment.segment_ordinal),
                        bucket_ordinal=bucket_ordinal,
                        sender_offset_rows=sender_offset,
                        receiver_offset_rows=0,
                        row_count=int(current_rows),
                        byte_count=int(payload_slices[0].payload_byte_count) if payload_slices else 0,
                        packed_send_layout_id=str(segment.packed_send_layout_id),
                        canonical_receive_layout_id="",
                        payload_slices=payload_slices,
                    )
                )
                consumed += current_rows
                bucket_ordinal += 1
    transfer_layouts.sort(key=lambda item: (int(item.src_rank), int(item.dst_rank), int(item.segment_ordinal)))
    return tuple(transfer_layouts), all_tasks


def estimate_planning_quantum_rows_from_values(values: list[int] | tuple[int, ...]) -> int:
    positive = sorted(int(value) for value in values if int(value) > 0)
    if not positive:
        return 1
    minimum = positive[0]
    lower = 1
    while (lower << 1) <= minimum:
        lower <<= 1
    if lower == minimum:
        return lower
    upper = lower << 1
    lower_relative_gap = (float(minimum) / float(lower)) - 1.0
    upper_relative_gap = (float(upper) / float(minimum)) - 1.0
    if upper_relative_gap < lower_relative_gap:
        return upper
    return lower


def estimate_planning_quantum_rows_from_contexts(
    *,
    global_contexts: tuple[PhaseReadyContext, ...],
    phase: str,
) -> int:
    row_counts = [
        int(segment.row_count)
        for context in global_contexts
        for segment in context.outgoing_segments
        if str(segment.phase) == str(phase) and int(segment.src_rank) != int(segment.dst_rank) and int(segment.row_count) > 0
    ]
    return estimate_planning_quantum_rows_from_values(row_counts)


def _compact_transfer_layout_dict(layout: Any) -> dict[str, Any]:
    return {
        "transfer_key": str(layout.transfer_key),
        "bundle_id": str(layout.bundle_id),
        "phase": str(layout.phase),
        "src_rank": int(layout.src_rank),
        "dst_rank": int(layout.dst_rank),
        "segment_ordinal": int(layout.segment_ordinal),
        "sender_offset_rows": int(layout.sender_offset_rows),
        "receiver_offset_rows": int(layout.receiver_offset_rows),
        "row_count": int(layout.row_count),
        "byte_count": int(layout.byte_count),
        "packed_send_layout_id": str(layout.packed_send_layout_id),
        "canonical_receive_layout_id": str(layout.canonical_receive_layout_id),
        "atomic_submit": bool(layout.atomic_submit),
        "payload_roles": tuple(str(payload.tensor_role) for payload in layout.payloads),
    }


def _compact_bucket_task_dict(task: BucketTask) -> dict[str, Any]:
    return {
        "task_id": str(task.task_id),
        "bundle_id": str(task.bundle_id),
        "phase": str(task.phase),
        "src_rank": int(task.src_rank),
        "dst_rank": int(task.dst_rank),
        "segment_ordinal": int(task.segment_ordinal),
        "bucket_ordinal": int(task.bucket_ordinal),
        "sender_offset_rows": int(task.sender_offset_rows),
        "receiver_offset_rows": int(task.receiver_offset_rows),
        "row_count": int(task.row_count),
        "byte_count": int(task.byte_count),
    }


def _compact_wave_dict(wave: PlanWave) -> dict[str, Any]:
    return {
        "wave_id": int(wave.wave_id),
        "phase": str(wave.phase),
        "task_ids": tuple(str(task.task_id) for task in wave.bucket_tasks),
    }


def _compact_policy_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    compact = dict(diagnostics)
    compact.pop("wave_edges", None)
    return compact


def pack_phase_tasks(tasks: list[BucketTask], *, phase: str) -> tuple[PlanWave, ...]:
    wave_tasks: list[list[BucketTask]] = []
    used_outgoing_masks: list[int] = []
    used_incoming_masks: list[int] = []
    for task in tasks:
        src = int(task.src_rank)
        dst = int(task.dst_rank)
        src_mask = 1 << src
        dst_mask = 1 << dst
        placed = False
        for wave_index, (used_outgoing_mask, used_incoming_mask) in enumerate(zip(used_outgoing_masks, used_incoming_masks)):
            if (used_outgoing_mask & src_mask) or (used_incoming_mask & dst_mask):
                continue
            wave_tasks[wave_index].append(task)
            used_outgoing_masks[wave_index] = used_outgoing_mask | src_mask
            used_incoming_masks[wave_index] = used_incoming_mask | dst_mask
            placed = True
            break
        if placed:
            continue
        wave_tasks.append([task])
        used_outgoing_masks.append(src_mask)
        used_incoming_masks.append(dst_mask)
    waves: list[PlanWave] = []
    for wave_id, bucket_tasks in enumerate(wave_tasks):
        waves.append(PlanWave(wave_id=wave_id, phase=phase, bucket_tasks=tuple(bucket_tasks)))
    return tuple(waves)


def finalize_execution_plan(
    *,
    local_context: PhaseReadyContext,
    policy_name: str,
    policy_version: str,
    capabilities: PolicyCapabilities,
    bucket_rows: int,
    transfer_layouts: tuple[Any, ...],
    all_tasks: list[BucketTask],
    waves: tuple[PlanWave, ...],
    diagnostics: dict[str, Any],
) -> PhaseExecutionPlan:
    compact_transfer_layouts = [_compact_transfer_layout_dict(layout) for layout in transfer_layouts]
    compact_tasks = [_compact_bucket_task_dict(task) for task in all_tasks]
    compact_waves = [_compact_wave_dict(wave) for wave in waves]
    compact_diagnostics = _compact_policy_diagnostics(diagnostics)
    phase_key = stable_hash(
        {
            "plan_key": local_context.plan_key,
            "phase": local_context.phase,
            "bucket_rows": int(bucket_rows),
            "policy_name": policy_name,
            "transfer_layouts": compact_transfer_layouts,
            "tasks": compact_tasks,
            "waves": compact_waves,
        }
    )
    plan = PhaseExecutionPlan(
        plan_key=local_context.plan_key,
        phase=local_context.phase,
        policy_name=policy_name,
        policy_version=policy_version,
        control_mode=local_context.control_mode,
        execution_mode="phase_sync_wave",
        transport_mutation=True,
        is_shadow_only=False,
        future_hint_mode=local_context.p2_hint.hint_mode,
        root_rank=int(local_context.ep_group_root_rank),
        observation_digest=phase_key,
        plan_hash="",
        waves=waves,
        metrics={
            "bucket_rows": int(bucket_rows),
            "bucket_count": len(all_tasks),
            "wave_count": len(waves),
            "phase": local_context.phase,
            "transport_mutation": True,
            "policy_name": policy_name,
            "policy_capabilities": capabilities.to_dict(),
            "uses_p2": capabilities.uses_p2,
            "p2_hint_seen": local_context.p2_hint.hint_mode != "none",
            "p2_influenced_plan": bool(diagnostics.get("p2_forecast_used", False)),
            "transfer_layouts": compact_transfer_layouts,
            "policy_diagnostics": compact_diagnostics,
            **compact_diagnostics,
        },
    )
    plan = replace(
        plan,
        plan_hash=stable_hash(
            {
                "plan_key": plan.plan_key,
                "phase": plan.phase,
                "policy_name": plan.policy_name,
                "policy_version": plan.policy_version,
                "control_mode": plan.control_mode,
                "execution_mode": plan.execution_mode,
                "transport_mutation": plan.transport_mutation,
                "future_hint_mode": plan.future_hint_mode,
                "root_rank": plan.root_rank,
                "observation_digest": plan.observation_digest,
                "waves": compact_waves,
            }
        ),
    )
    validate_phase_execution_plan(local_context, plan)
    return plan


def flows_from_matrix(matrix: tuple[tuple[int, ...], ...], *, phase: str, release_state: str, executable: bool) -> tuple[FlowDemand, ...]:
    flows: list[FlowDemand] = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=int(src_rank),
                    dst_rank=int(dst_rank),
                    byte_count=int(byte_count),
                    release_state=release_state,
                    is_executable=executable,
                )
            )
    return tuple(flows)


def phase_barrier_flow_window(problem: MultiPhaseSchedulingProblem) -> FlowWindow:
    return FlowWindow(
        ready_flows=flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
        blocked_flows=flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="blocked", executable=False),
        forecast_pressure=problem.flow_window.forecast_pressure,
    )


def pack_logical_waves(flows: list[FlowDemand], *, start_wave_id: int) -> tuple[LogicalWave, ...]:
    waves: list[LogicalWave] = []
    pending = list(flows)
    wave_id = start_wave_id
    while pending:
        used_outgoing: set[int] = set()
        used_incoming: set[int] = set()
        chosen: list[FlowDemand] = []
        remaining: list[FlowDemand] = []
        for flow in pending:
            if flow.src_rank in used_outgoing or flow.dst_rank in used_incoming:
                remaining.append(flow)
                continue
            chosen.append(flow)
            used_outgoing.add(flow.src_rank)
            used_incoming.add(flow.dst_rank)
        duration = float(max((flow.byte_count for flow in chosen), default=0))
        waves.append(LogicalWave(wave_id=wave_id, flows=tuple(chosen), duration=duration))
        pending = remaining
        wave_id += 1
    return tuple(waves)


def build_logical_plan_from_order(
    *,
    policy_name: str,
    policy_version: str,
    capabilities: PolicyCapabilities,
    ordered_p0: list[FlowDemand],
    ordered_p1: list[FlowDemand],
    ordered_p2: list[FlowDemand] | None = None,
    information_mode: str,
    p2_source: str,
    evaluation_eligible: bool,
    priority_components: dict[str, Any],
    tie_break_rule: str,
    fallback_reason: str = "",
) -> LogicalSchedulePlan:
    p0_waves = pack_logical_waves(ordered_p0, start_wave_id=0)
    p1_waves = pack_logical_waves(ordered_p1, start_wave_id=len(p0_waves))
    ordered_p2 = list(ordered_p2 or [])
    p2_waves = pack_logical_waves(ordered_p2, start_wave_id=len(p0_waves) + len(p1_waves))
    waves = tuple(p0_waves + p1_waves + p2_waves)
    per_wave: list[WaveDiagnostics] = []
    for wave in waves:
        total = float(sum(int(flow.byte_count) for flow in wave.flows))
        per_wave.append(
            WaveDiagnostics(
                wave_id=int(wave.wave_id),
                selected_flow_ids=tuple(flow.flow_id for flow in wave.flows),
                selected_edges=tuple(
                    {
                        "phase": flow.phase,
                        "src_rank": int(flow.src_rank),
                        "dst_rank": int(flow.dst_rank),
                        "byte_count": int(flow.byte_count),
                    }
                    for flow in wave.flows
                ),
                matching_weight=total,
                priority_components=priority_components,
                remaining_bytes_before=total,
                remaining_bytes_after=0.0,
                ready_flow_count_before=len(ordered_p0),
                blocked_flow_count_before=len(ordered_p1),
                forecast_pressure_summary={"source": p2_source},
                selection_reason=tie_break_rule,
            )
        )
    diag = PolicyDiagnostics(
        policy_name=policy_name,
        policy_version=policy_version,
        information_mode=information_mode,
        tie_break_rule=tie_break_rule,
        wave_count=len(waves),
        logical_flow_count=len(ordered_p0) + len(ordered_p1) + len(ordered_p2),
        ready_flow_count=len(ordered_p0),
        blocked_flow_count=len(ordered_p1),
        forecast_flow_count=len(ordered_p2),
        p1_dependency_used=capabilities.uses_blocked_p1_dependency,
        p2_forecast_used=capabilities.uses_p2_forecast,
        p2_source=p2_source,
        evaluation_eligible=evaluation_eligible,
        per_wave=tuple(per_wave),
        priority_components=priority_components,
        fallback_reason=fallback_reason,
    )
    return LogicalSchedulePlan(
        policy_name=policy_name,
        waves=waves,
        diagnostics=diag.to_dict(),
    )


def include_real_p2_phase(problem: MultiPhaseSchedulingProblem) -> bool:
    return problem.options.scheduling_mode == EXECUTION_WINDOW_MODE


def phase_local_future_information_mode(problem: MultiPhaseSchedulingProblem) -> str:
    if include_real_p2_phase(problem):
        return "oracle_execution_window"
    return "none"


def phase_local_evaluation_eligible(problem: MultiPhaseSchedulingProblem) -> bool:
    return not include_real_p2_phase(problem)


def build_phase_serial_release_aware_plan(
    *,
    problem: MultiPhaseSchedulingProblem,
    policy_name: str,
    policy_version: str,
    capabilities: PolicyCapabilities,
    information_mode: str,
    tie_break_rule: str,
    priority_components: dict[str, Any],
    p0_waves: tuple[LogicalWave, ...],
    p1_waves: tuple[LogicalWave, ...],
    p2_waves: tuple[LogicalWave, ...] = (),
    service_model: str = "phase_serial_wave",
    fallback_reason: str = "",
) -> LogicalSchedulePlan:
    waves = tuple(p0_waves + p1_waves + p2_waves)
    raw_schedule: list[dict[str, Any]] = []
    current_time = 0.0
    phase_index = {"p0_dispatch": 0, "p1_return": 1, "p2_next_dispatch": 2}
    for wave in waves:
        duration = float(wave.duration)
        start = current_time
        end = current_time + duration
        for flow in wave.flows:
            raw_schedule.append(
                {
                    "wave_id": int(wave.wave_id),
                    "phase": int(phase_index[flow.phase]),
                    "src_gpu": int(flow.src_rank),
                    "dst_gpu": int(flow.dst_rank),
                    "flow_id": str(flow.dependency_metadata.get("origin_flow_id", flow.flow_id)),
                    "chunk_id": str(flow.flow_id),
                    "served_volume": float(flow.byte_count),
                    "size": float(flow.byte_count),
                    "start": start,
                    "end": end,
                    "residual_before": float(flow.dependency_metadata.get("residual_before", flow.byte_count)),
                    "residual_after": float(flow.dependency_metadata.get("residual_after", 0.0)),
                    "priority": flow.dependency_metadata.get("priority", []),
                }
            )
        current_time = end
    audit = replay_and_audit_schedule(
        schedule=raw_schedule,
        dispatch_matrix=[list(row) for row in problem.p0_dispatch_matrix],
        combine_matrix=[list(row) for row in problem.p1_return_matrix],
        next_dispatch_matrix=[list(row) for row in problem.p2_next_dispatch_forecast_matrix],
        num_gpus=int(problem.topology.num_gpus),
        expert_compute_delay=float(problem.release_model.expert_compute_delay),
        mode=problem.options.scheduling_mode,
        scheduler_name=policy_name,
        planning_time_ms=0.0,
        reported_makespan=current_time,
        prediction_used=False,
    )
    diag = PolicyDiagnostics(
        policy_name=policy_name,
        policy_version=policy_version,
        information_mode=information_mode,
        tie_break_rule=tie_break_rule,
        wave_count=len(waves),
        logical_flow_count=sum(len(wave.flows) for wave in waves),
        ready_flow_count=len(problem.flow_window.ready_flows),
        blocked_flow_count=len(problem.flow_window.blocked_flows),
        forecast_flow_count=len(problem.flow_window.forecast_pressure),
        p1_dependency_used=False,
        p2_forecast_used=False,
        p2_source=problem.forecast.source if problem.forecast is not None else "none",
        evaluation_eligible=phase_local_evaluation_eligible(problem),
        per_wave=tuple(
            WaveDiagnostics(
                wave_id=int(wave.wave_id),
                selected_flow_ids=tuple(flow.flow_id for flow in wave.flows),
                selected_edges=tuple(
                    {
                        "phase": flow.phase,
                        "src_rank": int(flow.src_rank),
                        "dst_rank": int(flow.dst_rank),
                        "byte_count": int(flow.byte_count),
                        "origin_flow_id": flow.dependency_metadata.get("origin_flow_id", flow.flow_id),
                    }
                    for flow in wave.flows
                ),
                matching_weight=float(sum(int(flow.byte_count) for flow in wave.flows)),
                priority_components=priority_components,
                remaining_bytes_before=float(sum(float(flow.dependency_metadata.get("residual_before", flow.byte_count)) for flow in wave.flows)),
                remaining_bytes_after=float(sum(float(flow.dependency_metadata.get("residual_after", 0.0)) for flow in wave.flows)),
                ready_flow_count_before=len(wave.flows),
                blocked_flow_count_before=0,
                forecast_pressure_summary={"source": problem.forecast.source if problem.forecast is not None else "none"},
                selection_reason=tie_break_rule,
            )
            for wave in waves
        ),
        priority_components=priority_components,
        fallback_reason=fallback_reason,
    )
    diagnostics = {
        **diag.to_dict(),
        "policy_name": policy_name,
        "logical_model": "phase_serial_release_aware_replay",
        "service_model": service_model,
        "mode": problem.options.scheduling_mode,
        "future_information_mode": phase_local_future_information_mode(problem),
        "p2_role": "executable_actual_traffic" if include_real_p2_phase(problem) else "advisory_forecast_pressure",
        "p2_source": problem.forecast.source if problem.forecast is not None else "none",
        "forecast_available": problem.forecast is not None,
        "forecast_source": problem.forecast.source if problem.forecast is not None else "none",
        "forecast_consumed": False,
        "prediction_used": False,
        "evaluation_eligible": phase_local_evaluation_eligible(problem),
        "makespan": float(current_time),
        "logical_service_horizon": float(current_time),
        "planning_time_ms": 0.0,
        "solver_status": "valid" if audit.get("valid", False) else "invalid",
        "valid": bool(audit.get("valid", False)),
        "release_barrier_verified": not any("release violation" in error or "barrier violation" in error for error in audit.get("validation_errors", [])),
        "flow_conservation_verified": not any("volume mismatch" in error or "unexpected flow" in error for error in audit.get("validation_errors", [])),
        "matching_legality_verified": not any("overlap" in error for error in audit.get("validation_errors", [])),
        "online_executor_compatible": bool(capabilities.supports_online_phase_local_execution),
        "runtime_latency_comparable": False,
        "audit": {**audit, "planning_time_ms": 0.0},
        "raw_schedule": raw_schedule,
    }
    return LogicalSchedulePlan(policy_name=policy_name, waves=waves, diagnostics=diagnostics)


def plan_hash_from_waves(policy_name: str, waves: tuple[LogicalWave, ...]) -> str:
    payload = {
        "policy_name": policy_name,
        "waves": [wave.to_dict() for wave in waves],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def matching_summary(tasks: tuple[BucketTask, ...]) -> list[dict[str, Any]]:
    return [
        {
            "src_rank": int(task.src_rank),
            "dst_rank": int(task.dst_rank),
            "bucket_id": task.task_id,
            "byte_count": int(task.byte_count),
        }
        for task in tasks
    ]
