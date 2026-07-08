from __future__ import annotations

from rs.scheduling.contracts import LogicalSchedulePlan, MultiPhaseSchedulingProblem
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext

from ..capabilities import PolicyCapabilities
from .common import (
    build_phase_serial_release_aware_plan,
    build_logical_plan_from_order,
    build_transfer_layouts_and_tasks,
    fifo_task_key,
    finalize_execution_plan,
    flows_from_matrix,
    include_real_p2_phase,
    pack_phase_tasks,
)


class PhaseBarrierFIFOPolicy:
    policy_name = "phase_barrier_fifo"
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

    def __init__(self, *, bucket_rows: int, reported_policy_name: str | None = None) -> None:
        self.bucket_rows = int(bucket_rows)
        self.reported_policy_name = str(reported_policy_name or self.policy_name)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        ordered_p0 = sorted(
            flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
            key=lambda flow: (int(flow.src_rank), int(flow.dst_rank), str(flow.flow_id)),
        )
        ordered_p1 = sorted(
            flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True),
            key=lambda flow: (int(flow.src_rank), int(flow.dst_rank), str(flow.flow_id)),
        )
        ordered_p2 = []
        if include_real_p2_phase(problem):
            ordered_p2 = sorted(
                flows_from_matrix(problem.p2_next_dispatch_forecast_matrix, phase="p2_next_dispatch", release_state="ready", executable=True),
                key=lambda flow: (int(flow.src_rank), int(flow.dst_rank), str(flow.flow_id)),
            )
        base_plan = build_logical_plan_from_order(
            policy_name=self.reported_policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            ordered_p0=list(ordered_p0),
            ordered_p1=list(ordered_p1),
            ordered_p2=list(ordered_p2),
            information_mode="phase_barrier",
            p2_source=problem.forecast.source if problem.forecast is not None else "none",
            evaluation_eligible=True,
            priority_components={"sort_key": "src_rank,dst_rank,flow_id"},
            tie_break_rule="src_rank,dst_rank,flow_id",
        )
        p0_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p0_dispatch" for flow in wave.flows))
        p1_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p1_return" for flow in wave.flows))
        p2_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p2_next_dispatch" for flow in wave.flows))
        return build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.reported_policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="phase_barrier",
            tie_break_rule="src_rank,dst_rank,flow_id",
            priority_components={"sort_key": "src_rank,dst_rank,flow_id"},
            p0_waves=p0_waves,
            p1_waves=p1_waves,
            p2_waves=p2_waves,
            service_model="phase_serial_atomic",
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
        all_tasks.sort(key=fifo_task_key)
        waves = pack_phase_tasks(all_tasks, phase=local_context.phase)
        diagnostics = {
            "policy_name": self.reported_policy_name,
            "policy_version": self.policy_version,
            "information_mode": "phase_barrier",
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
            "priority_components": {"sort_key": "src_rank,dst_rank,segment_ordinal,bucket_ordinal"},
            "tie_break_rule": "src_rank,dst_rank,segment_ordinal,bucket_ordinal",
            "fallback_reason": "",
        }
        return finalize_execution_plan(
            local_context=local_context,
            policy_name=self.reported_policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            transfer_layouts=transfer_layouts,
            all_tasks=all_tasks,
            waves=waves,
            diagnostics=diagnostics,
        )


class BucketedFIFOPolicy(PhaseBarrierFIFOPolicy):
    def __init__(self, *, bucket_rows: int) -> None:
        super().__init__(bucket_rows=bucket_rows, reported_policy_name="bucketed_fifo")
