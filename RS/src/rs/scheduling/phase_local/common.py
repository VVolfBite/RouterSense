from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from typing import Any, Callable

from rs.scheduling.contracts import (
    FlowDemand,
    FlowWindow,
    LogicalSchedulePlan,
    LogicalWave,
    MultiPhaseSchedulingProblem,
)
from rs.scheduling.diagnostics import PolicyDiagnostics, WaveDiagnostics
from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.phase_execution_utils import bucketize_transfer_layouts
from rs.scheduling.phase_execution_utils import join_transfer_layouts
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
    transfer_layouts = join_transfer_layouts(global_contexts=global_contexts, phase=local_context.phase)
    remote_transfer_layouts = tuple(layout for layout in transfer_layouts if int(layout.src_rank) != int(layout.dst_rank))
    all_tasks = list(bucketize_transfer_layouts(remote_transfer_layouts, bucket_rows=int(bucket_rows)))
    return transfer_layouts, all_tasks


def pack_phase_tasks(tasks: list[BucketTask], *, phase: str) -> tuple[PlanWave, ...]:
    waves: list[PlanWave] = []
    pending = list(tasks)
    wave_id = 0
    while pending:
        used_outgoing: set[int] = set()
        used_incoming: set[int] = set()
        selected: list[BucketTask] = []
        remaining: list[BucketTask] = []
        for task in pending:
            src = int(task.src_rank)
            dst = int(task.dst_rank)
            if src in used_outgoing or dst in used_incoming:
                remaining.append(task)
                continue
            selected.append(task)
            used_outgoing.add(src)
            used_incoming.add(dst)
        waves.append(PlanWave(wave_id=wave_id, phase=phase, bucket_tasks=tuple(selected)))
        pending = remaining
        wave_id += 1
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
    phase_key = stable_hash(
        {
            "plan_key": local_context.plan_key,
            "phase": local_context.phase,
            "bucket_rows": int(bucket_rows),
            "policy_name": policy_name,
            "transfer_layouts": [layout.to_dict() for layout in transfer_layouts],
            "tasks": [task.to_dict() for task in all_tasks],
            "waves": [wave.to_dict() for wave in waves],
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
            "transfer_layouts": [layout.to_dict() for layout in transfer_layouts],
            "policy_diagnostics": diagnostics,
            **diagnostics,
        },
    )
    plan = replace(plan, plan_hash=stable_hash(plan.to_dict()))
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
    information_mode: str,
    p2_source: str,
    evaluation_eligible: bool,
    priority_components: dict[str, Any],
    tie_break_rule: str,
    fallback_reason: str = "",
) -> LogicalSchedulePlan:
    p0_waves = pack_logical_waves(ordered_p0, start_wave_id=0)
    p1_waves = pack_logical_waves(ordered_p1, start_wave_id=len(p0_waves))
    waves = tuple(p0_waves + p1_waves)
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
        logical_flow_count=len(ordered_p0) + len(ordered_p1),
        ready_flow_count=len(ordered_p0),
        blocked_flow_count=len(ordered_p1),
        forecast_flow_count=0,
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
