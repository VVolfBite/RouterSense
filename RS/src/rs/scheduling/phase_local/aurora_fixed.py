from __future__ import annotations

import time
from collections import defaultdict

from rs.scheduling.contracts import LogicalSchedulePlan, MultiPhaseSchedulingProblem
from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave

from .common import build_logical_plan_from_order, build_transfer_layouts_and_tasks, finalize_execution_plan, flows_from_matrix
from .common import build_phase_serial_release_aware_plan, include_real_p2_phase
from ..capabilities import PolicyCapabilities


class AuroraOrderFixedPolicy:
    policy_name = "aurora_order_fixed"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=True,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=False,
        uses_p2_forecast=False,
        requires_fixed_placement=True,
        evaluation_eligible=True,
    )

    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        def order_phase(flows):
            remaining = list(flows)
            ordered = []
            while remaining:
                source_pressure = defaultdict(int)
                destination_pressure = defaultdict(int)
                for flow in remaining:
                    source_pressure[int(flow.src_rank)] += int(flow.byte_count)
                    destination_pressure[int(flow.dst_rank)] += int(flow.byte_count)
                remaining.sort(
                    key=lambda flow: (
                        -max(source_pressure[int(flow.src_rank)], destination_pressure[int(flow.dst_rank)]),
                        -int(flow.byte_count),
                        int(flow.src_rank),
                        int(flow.dst_rank),
                        str(flow.flow_id),
                    )
                )
                ordered.append(remaining.pop(0))
            return ordered

        ordered_p0 = order_phase(flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True))
        ordered_p1 = order_phase(flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True))
        ordered_p2 = []
        if include_real_p2_phase(problem):
            ordered_p2 = order_phase(
                flows_from_matrix(problem.p2_next_dispatch_forecast_matrix, phase="p2_next_dispatch", release_state="ready", executable=True)
            )
        base_plan = build_logical_plan_from_order(
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            ordered_p0=ordered_p0,
            ordered_p1=ordered_p1,
            ordered_p2=ordered_p2,
            information_mode="phase_local_pressure",
            p2_source=problem.forecast.source if problem.forecast is not None else "none",
            evaluation_eligible=True,
            priority_components={"priority": "max(source_pressure,destination_pressure)->flow bytes->src->dst->flow_id"},
            tie_break_rule="src_rank,dst_rank,flow_id",
        )
        p0_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p0_dispatch" for flow in wave.flows))
        p1_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p1_return" for flow in wave.flows))
        p2_waves = tuple(wave for wave in base_plan.waves if all(flow.phase == "p2_next_dispatch" for flow in wave.flows))
        return build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="phase_local_pressure",
            tie_break_rule="src_rank,dst_rank,flow_id",
            priority_components={"priority": "max(source_pressure,destination_pressure)->flow bytes->src->dst->flow_id"},
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
        transfer_layouts, all_tasks, build_stats = build_transfer_layouts_and_tasks(
            local_context=local_context,
            global_contexts=global_contexts,
            bucket_rows=self.bucket_rows,
            return_stats=True,
        )
        schedule_started_ns = time.perf_counter_ns()
        remaining = list(all_tasks)
        waves: list[PlanWave] = []
        wave_scores: list[float] = []
        selection_trace: list[dict] = []
        wave_id = 0
        while remaining:
            source_pressure = defaultdict(int)
            destination_pressure = defaultdict(int)
            for task in remaining:
                source_pressure[int(task.src_rank)] += int(task.byte_count)
                destination_pressure[int(task.dst_rank)] += int(task.byte_count)
            used_outgoing: set[int] = set()
            used_incoming: set[int] = set()
            selected: list[BucketTask] = []
            wave_weight = 0.0
            while True:
                candidates = [task for task in remaining if int(task.src_rank) not in used_outgoing and int(task.dst_rank) not in used_incoming]
                if not candidates:
                    break
                candidates.sort(
                    key=lambda task: (
                        -max(source_pressure[int(task.src_rank)], destination_pressure[int(task.dst_rank)]),
                        -int(task.byte_count),
                        int(task.src_rank),
                        int(task.dst_rank),
                        int(task.bucket_ordinal),
                    )
                )
                chosen = candidates[0]
                src = int(chosen.src_rank)
                dst = int(chosen.dst_rank)
                score = float(max(source_pressure[src], destination_pressure[dst]))
                selection_trace.append(
                    {
                        "wave_id": wave_id,
                        "bucket_id": chosen.task_id,
                        "src_rank": src,
                        "dst_rank": dst,
                        "source_pressure": int(source_pressure[src]),
                        "destination_pressure": int(destination_pressure[dst]),
                        "priority_score": score,
                        "selection_reason": "max(source_pressure,destination_pressure)->byte_count->src->dst->bucket",
                    }
                )
                selected.append(chosen)
                wave_weight += float(chosen.byte_count)
                used_outgoing.add(src)
                used_incoming.add(dst)
                remaining.remove(chosen)
                source_pressure[src] -= int(chosen.byte_count)
                destination_pressure[dst] -= int(chosen.byte_count)
            waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(selected)))
            wave_scores.append(wave_weight)
            wave_id += 1
        pack_time_us = (time.perf_counter_ns() - schedule_started_ns) / 1000.0

        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_capabilities": self.capabilities.to_dict(),
            "bucket_order": [task.task_id for wave in waves for task in wave.bucket_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": wave_scores,
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "priority_components": {
                "source_pressure": "remaining outgoing bytes at source",
                "destination_pressure": "remaining incoming bytes at destination",
                "priority": "max(source_pressure,destination_pressure)->flow bytes",
                "selection_trace": selection_trace,
            },
            "tie_break_rule": "src_rank,dst_rank,bucket_ordinal",
            "fallback_reason": "",
            "evaluation_eligible": True,
        }
        return finalize_execution_plan(
            local_context=local_context,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            transfer_layouts=transfer_layouts,
            all_tasks=[task for wave in waves for task in wave.bucket_tasks],
            waves=tuple(waves),
            diagnostics=diagnostics,
            timing_metrics={
                **build_stats,
                "pack_phase_tasks_time_us": pack_time_us,
                "wave_count": int(len(waves)),
                "max_wave_task_count": int(max((len(wave.bucket_tasks) for wave in waves), default=0)),
                "task_count": int(len(all_tasks)),
            },
        )
