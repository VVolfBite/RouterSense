"""Greedy max-weight decomposition style reference.

This is an offline/reference implementation of the paper's decomposition core:
operate directly on the residual MoE traffic matrix, extract a maximum-weight
matching, subtract the largest common service quantum, and repeat.  It does not
model the paper's photonic reconfiguration or expert-compute cost curves.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.matching import maximum_weight_bipartite_matching
from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.phase_local.common import (
    build_phase_serial_release_aware_plan,
    build_transfer_layouts_and_tasks,
    finalize_execution_plan,
    flows_from_matrix,
    include_real_p2_phase,
)


class GMWDStylePolicy:
    policy_name = "gmwd_style_reference"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=False,
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
        ranks = tuple(range(int(problem.topology.num_gpus)))
        p0_waves, p0_trace = _decompose_phase(
            flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
            ranks=ranks,
            start_wave_id=0,
        )
        p1_waves, p1_trace = _decompose_phase(
            flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True),
            ranks=ranks,
            start_wave_id=len(p0_waves),
        )
        p2_waves: list[LogicalWave] = []
        p2_trace: list[dict[str, Any]] = []
        if include_real_p2_phase(problem):
            p2_waves, p2_trace = _decompose_phase(
                flows_from_matrix(
                    problem.p2_next_dispatch_forecast_matrix,
                    phase="p2_next_dispatch",
                    release_state="ready",
                    executable=True,
                ),
                ranks=ranks,
                start_wave_id=len(p0_waves) + len(p1_waves),
            )
        trace = p0_trace + p1_trace + p2_trace
        base = build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="phase_local_gmwd_style",
            tie_break_rule="maximum matching weight then lexicographic edge order",
            priority_components={
                "selection_rule": "maximum-weight residual matching",
                "service_rule": "minimum selected residual quantum",
            },
            p0_waves=tuple(p0_waves),
            p1_waves=tuple(p1_waves),
            p2_waves=tuple(p2_waves),
            service_model="phase_serial_gmwd_fluid_v1",
        )
        return LogicalSchedulePlan(
            policy_name=base.policy_name,
            waves=base.waves,
            diagnostics={
                **base.diagnostics,
                "literature_mapping": "style",
                "implemented_core": (
                    "residual_matrix",
                    "maximum_weight_matching",
                    "common_quantum_subtraction",
                ),
                "missing_mechanisms": (
                    "photonic_reconfiguration_cost",
                    "profiled_expert_compute_cost",
                    "communication_compute_overlap_objective",
                ),
                "gmwd_trace": trace,
                "matching_count": len(trace),
                "mean_selected_batch": (
                    float(sum(int(item["selected_volume"]) for item in trace)) / float(len(trace)) if trace else 0.0
                ),
            },
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
        edge_queues: dict[tuple[int, int], list[BucketTask]] = defaultdict(list)
        for task in all_tasks:
            edge_queues[(int(task.src_rank), int(task.dst_rank))].append(task)
        for queue in edge_queues.values():
            queue.sort(key=lambda task: (int(task.segment_ordinal), int(task.bucket_ordinal), str(task.task_id)))
        ranks = tuple(int(rank) for rank in local_context.ep_group_ranks)
        waves: list[PlanWave] = []
        trace: list[dict[str, Any]] = []
        while any(edge_queues.values()):
            residual = {
                edge: sum(int(task.byte_count) for task in queue)
                for edge, queue in edge_queues.items()
                if queue
            }
            chosen_edges = maximum_weight_bipartite_matching(
                sources=ranks,
                destinations=ranks,
                edge_weight=lambda src, dst: float(residual.get((src, dst), 0)),
            )
            chosen_tasks = [edge_queues[edge].pop(0) for edge in chosen_edges if edge_queues.get(edge)]
            if not chosen_tasks:
                raise RuntimeError("gmwd_style_reference made no progress")
            wave_id = len(waves)
            waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(chosen_tasks)))
            trace.append(
                {
                    "wave_id": wave_id,
                    "selected_edges": [
                        {
                            "src_rank": int(task.src_rank),
                            "dst_rank": int(task.dst_rank),
                            "bucket_id": str(task.task_id),
                            "byte_count": int(task.byte_count),
                        }
                        for task in chosen_tasks
                    ],
                    "matching_weight": int(sum(residual[(int(task.src_rank), int(task.dst_rank))] for task in chosen_tasks)),
                    "selected_volume": int(sum(int(task.byte_count) for task in chosen_tasks)),
                }
            )
        pack_time_us = (time.perf_counter_ns() - schedule_started_ns) / 1000.0
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "information_mode": "phase_local_gmwd_style",
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "evaluation_eligible": True,
            "gmwd_trace": trace,
            "priority_components": {
                "selection_rule": "maximum residual bytes per matching",
                "service_unit": "one canonical bucket per selected edge",
            },
            "tie_break_rule": "maximum matching weight then stable task order",
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
            waves=tuple(waves),
            diagnostics=diagnostics,
            timing_metrics={
                **build_stats,
                "pack_phase_tasks_time_us": pack_time_us,
                "wave_count": len(waves),
                "task_count": len(all_tasks),
            },
        )


def _decompose_phase(
    flows: tuple[FlowDemand, ...],
    *,
    ranks: tuple[int, ...],
    start_wave_id: int,
) -> tuple[list[LogicalWave], list[dict[str, Any]]]:
    by_edge = {(int(flow.src_rank), int(flow.dst_rank)): flow for flow in flows}
    residual = {edge: int(flow.byte_count) for edge, flow in by_edge.items() if int(flow.byte_count) > 0}
    waves: list[LogicalWave] = []
    trace: list[dict[str, Any]] = []
    wave_id = int(start_wave_id)
    while residual:
        selected_edges = tuple(
            edge
            for edge in maximum_weight_bipartite_matching(
                sources=ranks,
                destinations=ranks,
                edge_weight=lambda src, dst: float(residual.get((src, dst), 0)),
            )
            if edge in residual
        )
        if not selected_edges:
            raise RuntimeError("gmwd_style_reference made no progress on residual matrix")
        quantum = min(int(residual[edge]) for edge in selected_edges)
        selected_flows: list[FlowDemand] = []
        residual_before_total = int(sum(residual.values()))
        for edge in selected_edges:
            original = by_edge[edge]
            before = int(residual[edge])
            after = before - int(quantum)
            selected_flows.append(
                FlowDemand(
                    flow_id=f"{original.flow_id}:gmwd:{wave_id}",
                    phase=original.phase,
                    src_rank=int(original.src_rank),
                    dst_rank=int(original.dst_rank),
                    byte_count=int(quantum),
                    release_state="ready",
                    is_executable=True,
                    dependency_metadata={
                        "origin_flow_id": original.flow_id,
                        "residual_before": before,
                        "residual_after": after,
                        "priority": [float(before)],
                    },
                )
            )
            if after > 0:
                residual[edge] = after
            else:
                residual.pop(edge, None)
        waves.append(LogicalWave(wave_id=wave_id, flows=tuple(selected_flows), duration=float(quantum)))
        trace.append(
            {
                "wave_id": wave_id,
                "selected_edges": [
                    {"src_rank": src, "dst_rank": dst, "residual_before": int(by_edge[(src, dst)].byte_count)}
                    for src, dst in selected_edges
                ],
                "service_quantum": int(quantum),
                "matching_weight": int(sum(int(flow.dependency_metadata["residual_before"]) for flow in selected_flows)),
                "selected_volume": int(sum(int(flow.byte_count) for flow in selected_flows)),
                "residual_total_before": residual_before_total,
                "residual_total_after": int(sum(residual.values())),
            }
        )
        wave_id += 1
    return waves, trace


__all__ = ["GMWDStylePolicy"]
