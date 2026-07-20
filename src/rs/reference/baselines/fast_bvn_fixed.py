"""FAST-style two-tier stage scheduling reference.

The reference captures FAST's core scheduling abstraction under RouterSense's
fixed endpoint contract: collapse GPU traffic to server pairs, keep bottleneck
servers active with maximum-weight one-to-one server stages, then realize each
stage with conflict-free GPU transfers while opportunistically filling idle
ports with intra-server traffic. It does not mutate endpoints, so explicit
intra-server rebalance/redistribution is represented only in diagnostics.
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


class FastBVNSingleTierPolicy:
    # The class name is retained for source compatibility; the implementation is
    # now a two-tier FAST-style core rather than the old single-tier ordering.
    policy_name = "fast_stage_reference"
    policy_version = "v2-two-tier-style"
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

    def __init__(self, *, bucket_rows: int, gpus_per_server: int = 0) -> None:
        self.bucket_rows = int(bucket_rows)
        self.gpus_per_server = int(gpus_per_server)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        rank_count = int(problem.topology.num_gpus)
        ranks = tuple(range(rank_count))
        gpus_per_server = _infer_gpus_per_server(rank_count, self.gpus_per_server)
        p0_waves, p0_trace = _schedule_logical_phase(
            flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
            ranks=ranks,
            gpus_per_server=gpus_per_server,
            start_wave_id=0,
        )
        p1_waves, p1_trace = _schedule_logical_phase(
            flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True),
            ranks=ranks,
            gpus_per_server=gpus_per_server,
            start_wave_id=len(p0_waves),
        )
        p2_waves: list[LogicalWave] = []
        p2_trace: list[dict[str, Any]] = []
        if include_real_p2_phase(problem):
            p2_waves, p2_trace = _schedule_logical_phase(
                flows_from_matrix(
                    problem.p2_next_dispatch_forecast_matrix,
                    phase="p2_next_dispatch",
                    release_state="ready",
                    executable=True,
                ),
                ranks=ranks,
                gpus_per_server=gpus_per_server,
                start_wave_id=len(p0_waves) + len(p1_waves),
            )
        trace = p0_trace + p1_trace + p2_trace
        base = build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="phase_local_fast_two_tier_style",
            tie_break_rule="maximum server-pair weight then maximum GPU-edge weight",
            priority_components={
                "server_stage": "maximum-weight one-to-one server matching",
                "gpu_realization": "maximum-weight conflict-free original endpoint matching",
                "local_fill": "fill unused GPU ports with intra-server transfers",
            },
            p0_waves=tuple(p0_waves),
            p1_waves=tuple(p1_waves),
            p2_waves=tuple(p2_waves),
            service_model="phase_serial_fast_two_tier_style_v2",
        )
        return LogicalSchedulePlan(
            policy_name=base.policy_name,
            waves=base.waves,
            diagnostics={
                **base.diagnostics,
                "literature_mapping": "style",
                "gpus_per_server": gpus_per_server,
                "server_count": (rank_count + gpus_per_server - 1) // gpus_per_server,
                "implemented_core": (
                    "server_level_traffic_collapse",
                    "balanced_one_to_one_server_stages",
                    "bottleneck_server_continuity",
                    "intra_server_idle_port_fill",
                ),
                "missing_mechanisms": (
                    "endpoint_mutating_intra_server_rebalance",
                    "per_stage_redistribution",
                    "scale_up_scale_out_pipeline_cost_model",
                ),
                "fast_stage_trace": trace,
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
        ranks = tuple(int(rank) for rank in local_context.ep_group_ranks)
        gpus_per_server = _infer_gpus_per_server(len(ranks), self.gpus_per_server)
        started_ns = time.perf_counter_ns()
        wave_tasks, trace = _schedule_bucket_tasks(
            all_tasks,
            ranks=ranks,
            gpus_per_server=gpus_per_server,
        )
        waves = tuple(
            PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(tasks))
            for wave_id, tasks in enumerate(wave_tasks)
        )
        pack_time_us = (time.perf_counter_ns() - started_ns) / 1000.0
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "uses_current_phase_demand": True,
            "uses_p1_reservation": False,
            "uses_p2_hint": False,
            "evaluation_eligible": True,
            "gpus_per_server": gpus_per_server,
            "fast_stage_trace": trace,
            "per_wave_matching_weight": [
                float(sum(int(task.byte_count) for task in tasks)) for tasks in wave_tasks
            ],
            "priority_components": {
                "server_stage": "maximum residual server-pair traffic",
                "gpu_realization": "maximum residual GPU-edge traffic",
            },
            "tie_break_rule": "maximum weight then stable edge/task order",
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
            timing_metrics={
                **build_stats,
                "pack_phase_tasks_time_us": pack_time_us,
                "wave_count": len(waves),
                "task_count": len(all_tasks),
            },
        )


def _infer_gpus_per_server(rank_count: int, requested: int) -> int:
    if rank_count <= 1:
        return 1
    if requested > 0:
        if rank_count % requested != 0:
            raise ValueError("gpus_per_server must divide rank_count")
        return requested
    for candidate in (4, 2):
        if rank_count > candidate and rank_count % candidate == 0:
            return candidate
    if rank_count == 4:
        return 2
    return 1


def _server(rank: int, gpus_per_server: int) -> int:
    return int(rank) // int(gpus_per_server)


def _rank_group(server_id: int, ranks: tuple[int, ...], gpus_per_server: int) -> tuple[int, ...]:
    return tuple(rank for rank in ranks if _server(rank, gpus_per_server) == server_id)


def _select_edges(
    residual: dict[tuple[int, int], int],
    *,
    ranks: tuple[int, ...],
    gpus_per_server: int,
) -> tuple[list[tuple[int, int]], tuple[tuple[int, int], ...]]:
    server_ids = tuple(sorted({_server(rank, gpus_per_server) for rank in ranks}))
    server_residual: dict[tuple[int, int], int] = defaultdict(int)
    for (src, dst), value in residual.items():
        src_server = _server(src, gpus_per_server)
        dst_server = _server(dst, gpus_per_server)
        if src_server != dst_server:
            server_residual[(src_server, dst_server)] += int(value)
    server_pairs = maximum_weight_bipartite_matching(
        sources=server_ids,
        destinations=server_ids,
        edge_weight=lambda src, dst: float(server_residual.get((src, dst), 0)) if src != dst else 0.0,
    )
    selected: list[tuple[int, int]] = []
    for src_server, dst_server in server_pairs:
        src_ranks = _rank_group(src_server, ranks, gpus_per_server)
        dst_ranks = _rank_group(dst_server, ranks, gpus_per_server)
        selected.extend(
            edge
            for edge in maximum_weight_bipartite_matching(
                sources=src_ranks,
                destinations=dst_ranks,
                edge_weight=lambda src, dst: float(residual.get((src, dst), 0)),
            )
            if edge in residual
        )
    used_src = {src for src, _ in selected}
    used_dst = {dst for _, dst in selected}
    local_candidates = {
        edge: value
        for edge, value in residual.items()
        if _server(edge[0], gpus_per_server) == _server(edge[1], gpus_per_server)
        and edge[0] not in used_src
        and edge[1] not in used_dst
    }
    if local_candidates:
        selected.extend(
            edge
            for edge in maximum_weight_bipartite_matching(
                sources=tuple(rank for rank in ranks if rank not in used_src),
                destinations=tuple(rank for rank in ranks if rank not in used_dst),
                edge_weight=lambda src, dst: float(local_candidates.get((src, dst), 0)),
            )
            if edge in local_candidates
        )
    if not selected and residual:
        selected = list(
            maximum_weight_bipartite_matching(
                sources=ranks,
                destinations=ranks,
                edge_weight=lambda src, dst: float(residual.get((src, dst), 0)),
            )
        )
    return selected, tuple(server_pairs)


def _schedule_logical_phase(
    flows: tuple[FlowDemand, ...],
    *,
    ranks: tuple[int, ...],
    gpus_per_server: int,
    start_wave_id: int,
) -> tuple[list[LogicalWave], list[dict[str, Any]]]:
    by_edge = {(int(flow.src_rank), int(flow.dst_rank)): flow for flow in flows}
    residual = {edge: int(flow.byte_count) for edge, flow in by_edge.items() if int(flow.byte_count) > 0}
    waves: list[LogicalWave] = []
    trace: list[dict[str, Any]] = []
    wave_id = int(start_wave_id)
    while residual:
        selected_edges, server_pairs = _select_edges(
            residual,
            ranks=ranks,
            gpus_per_server=gpus_per_server,
        )
        selected_edges = [edge for edge in selected_edges if edge in residual]
        if not selected_edges:
            raise RuntimeError("fast_stage_reference made no progress")
        quantum = min(int(residual[edge]) for edge in selected_edges)
        selected_flows: list[FlowDemand] = []
        before_total = int(sum(residual.values()))
        for edge in selected_edges:
            original = by_edge[edge]
            before = int(residual[edge])
            after = before - quantum
            selected_flows.append(
                FlowDemand(
                    flow_id=f"{original.flow_id}:fast:{wave_id}",
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
                        "src_server": _server(int(original.src_rank), gpus_per_server),
                        "dst_server": _server(int(original.dst_rank), gpus_per_server),
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
                "server_pairs": [
                    {"src_server": src, "dst_server": dst} for src, dst in server_pairs
                ],
                "gpu_edges": [
                    {
                        "src_rank": src,
                        "dst_rank": dst,
                        "src_server": _server(src, gpus_per_server),
                        "dst_server": _server(dst, gpus_per_server),
                    }
                    for src, dst in selected_edges
                ],
                "service_quantum": quantum,
                "residual_total_before": before_total,
                "residual_total_after": int(sum(residual.values())),
            }
        )
        wave_id += 1
    return waves, trace


def _schedule_bucket_tasks(
    tasks: list[BucketTask],
    *,
    ranks: tuple[int, ...],
    gpus_per_server: int,
) -> tuple[list[list[BucketTask]], list[dict[str, Any]]]:
    edge_queues: dict[tuple[int, int], list[BucketTask]] = defaultdict(list)
    for task in tasks:
        edge_queues[(int(task.src_rank), int(task.dst_rank))].append(task)
    for queue in edge_queues.values():
        queue.sort(key=lambda task: (int(task.segment_ordinal), int(task.bucket_ordinal), str(task.task_id)))
    waves: list[list[BucketTask]] = []
    trace: list[dict[str, Any]] = []
    while any(edge_queues.values()):
        residual = {
            edge: sum(int(task.byte_count) for task in queue)
            for edge, queue in edge_queues.items()
            if queue
        }
        selected_edges, server_pairs = _select_edges(
            residual,
            ranks=ranks,
            gpus_per_server=gpus_per_server,
        )
        selected_tasks = [edge_queues[edge].pop(0) for edge in selected_edges if edge_queues.get(edge)]
        if not selected_tasks:
            raise RuntimeError("fast_stage_reference made no progress on bucket tasks")
        wave_id = len(waves)
        waves.append(selected_tasks)
        trace.append(
            {
                "wave_id": wave_id,
                "server_pairs": [
                    {"src_server": src, "dst_server": dst} for src, dst in server_pairs
                ],
                "selected_tasks": [
                    {
                        "task_id": str(task.task_id),
                        "src_rank": int(task.src_rank),
                        "dst_rank": int(task.dst_rank),
                        "byte_count": int(task.byte_count),
                    }
                    for task in selected_tasks
                ],
            }
        )
    return waves, trace


__all__ = ["FastBVNSingleTierPolicy"]
