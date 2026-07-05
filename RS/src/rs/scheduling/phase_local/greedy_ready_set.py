from __future__ import annotations

from rs.scheduling.contracts import LogicalSchedulePlan, MultiPhaseSchedulingProblem
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext

from ..capabilities import PolicyCapabilities
from .common import (
    build_logical_plan_from_order,
    build_transfer_layouts_and_tasks,
    finalize_execution_plan,
    flows_from_matrix,
    pack_phase_tasks,
)


def _greedy_sort_key(flow_or_task) -> tuple[int, int, int, str]:
    return (-int(flow_or_task.byte_count), int(flow_or_task.src_rank), int(flow_or_task.dst_rank), str(getattr(flow_or_task, "flow_id", getattr(flow_or_task, "task_id", ""))))


class GreedyReadySetPolicy:
    policy_name = "greedy_ready_set"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=True,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=False,
        uses_p2_forecast=False,
        requires_fixed_placement=False,
        evaluation_eligible=True,
    )

    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        ordered_p0 = sorted(
            flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
            key=_greedy_sort_key,
        )
        ordered_p1 = sorted(
            flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True),
            key=_greedy_sort_key,
        )
        return build_logical_plan_from_order(
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            ordered_p0=list(ordered_p0),
            ordered_p1=list(ordered_p1),
            information_mode="ready_only",
            p2_source=problem.forecast.source if problem.forecast is not None else "none",
            evaluation_eligible=True,
            priority_components={"sort_key": "remaining_byte_count desc, src_rank, dst_rank, flow_id"},
            tie_break_rule="remaining_byte_count desc, src_rank, dst_rank, flow_id",
        )

    def build_plan(
        self,
        *,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> PhaseExecutionPlan:
        transfer_layouts, all_tasks = build_transfer_layouts_and_tasks(
            local_context=local_context,
            global_contexts=global_contexts,
            bucket_rows=self.bucket_rows,
        )
        all_tasks.sort(key=_greedy_sort_key)
        waves = pack_phase_tasks(all_tasks, phase=local_context.phase)
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "information_mode": "ready_only",
            "bucket_order": [task.task_id for task in all_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": [float(sum(int(task.byte_count) for task in wave.bucket_tasks)) for wave in waves],
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "p1_dependency_used": False,
            "p2_forecast_used": False,
            "p2_source": local_context.p2_hint.hint_source,
            "evaluation_eligible": True,
            "priority_components": {"sort_key": "byte_count desc, src_rank, dst_rank, task_id"},
            "tie_break_rule": "byte_count desc, src_rank, dst_rank, task_id",
            "fallback_reason": "",
        }
        return finalize_execution_plan(
            local_context=local_context,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            transfer_layouts=transfer_layouts,
            all_tasks=all_tasks,
            waves=waves,
            diagnostics=diagnostics,
        )
