from __future__ import annotations

from dataclasses import replace
from typing import Callable

from rs.scheduling.phase_execution_utils import bucketize_transfer_layouts
from rs.scheduling.phase_execution_utils import validate_phase_execution_plan
from rs.scheduling.phase_execution_utils import join_transfer_layouts, pack_waves
from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.validation import stable_hash

from .capabilities import PolicyCapabilities


def fifo_task_key(task: BucketTask) -> tuple[int, int, int, int]:
    return (int(task.src_rank), int(task.dst_rank), int(task.segment_ordinal), int(task.bucket_ordinal))


def reverse_bucket_task_key(task: BucketTask) -> tuple[int, int, int, int]:
    return (int(task.src_rank), int(task.dst_rank), -int(task.segment_ordinal), -int(task.bucket_ordinal))


def _pack_waves(tasks: list[BucketTask], *, phase: str) -> tuple[PlanWave, ...]:
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


def build_transfer_layouts_and_tasks(
    *,
    local_context: PhaseReadyContext,
    global_contexts: tuple[PhaseReadyContext, ...],
    bucket_rows: int,
) -> tuple[tuple, list[BucketTask]]:
    transfer_layouts = join_transfer_layouts(global_contexts=global_contexts, phase=local_context.phase)
    remote_transfer_layouts = tuple(
        layout for layout in transfer_layouts if int(layout.src_rank) != int(layout.dst_rank)
    )
    all_tasks = list(bucketize_transfer_layouts(remote_transfer_layouts, bucket_rows=int(bucket_rows)))
    return transfer_layouts, all_tasks


def finalize_execution_plan(
    *,
    local_context: PhaseReadyContext,
    policy_name: str,
    policy_version: str,
    capabilities: PolicyCapabilities,
    bucket_rows: int,
    transfer_layouts: tuple,
    all_tasks: list[BucketTask],
    waves: tuple[PlanWave, ...],
    diagnostics: dict,
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
            "p2_influenced_plan": False,
            "transfer_layouts": [layout.to_dict() for layout in transfer_layouts],
            "policy_diagnostics": diagnostics,
            **diagnostics,
        },
    )
    plan = replace(plan, plan_hash=stable_hash(plan.to_dict()))
    validate_phase_execution_plan(local_context, plan)
    return plan


def build_bucket_execution_plan(
    *,
    local_context: PhaseReadyContext,
    global_contexts: tuple[PhaseReadyContext, ...],
    policy_name: str,
    policy_version: str,
    capabilities: PolicyCapabilities,
    bucket_rows: int,
    task_sort_key: Callable[[BucketTask], tuple[int, int, int, int]],
) -> PhaseExecutionPlan:
    transfer_layouts, all_tasks = build_transfer_layouts_and_tasks(
        local_context=local_context,
        global_contexts=global_contexts,
        bucket_rows=bucket_rows,
    )
    all_tasks.sort(key=task_sort_key)
    waves = _pack_waves(all_tasks, phase=local_context.phase)
    diagnostics = {
        "bucket_order": [task.task_id for task in all_tasks],
        "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
        "per_wave_matching_weight": [float(sum(int(task.byte_count) for task in wave.bucket_tasks)) for wave in waves],
        "uses_current_phase_demand": True,
        "uses_p1_reservation": False,
        "uses_p2_hint": False,
        "priority_components": {"sort_key": "fifo(src_rank,dst_rank,segment_ordinal,bucket_ordinal)"},
        "tie_break_rule": "src_rank,dst_rank,segment_ordinal,bucket_ordinal",
        "fallback_reason": "",
        "evaluation_eligible": True,
    }
    return finalize_execution_plan(
        local_context=local_context,
        policy_name=policy_name,
        policy_version=policy_version,
        capabilities=capabilities,
        bucket_rows=bucket_rows,
        transfer_layouts=transfer_layouts,
        all_tasks=all_tasks,
        waves=waves,
        diagnostics=diagnostics,
    )


class BucketedFIFOPolicy:
    policy_name = "bucketed_fifo"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        uses_p0=True,
        uses_p1=True,
        uses_p2=False,
        cross_phase=False,
        requires_topology=False,
        supports_sync_before_phase=True,
        supports_default_continue=False,
    )

    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def build_plan(
        self,
        *,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> PhaseExecutionPlan:
        return build_bucket_execution_plan(
            local_context=local_context,
            global_contexts=global_contexts,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            task_sort_key=fifo_task_key,
        )
