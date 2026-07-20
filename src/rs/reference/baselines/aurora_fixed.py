"""Aurora-style fixed-placement transmission ordering reference.

This implementation isolates Aurora's Exclusive+Homogeneous communication core:
seed the schedule from the busiest sender, process remaining senders by traffic
volume, and place each transmission in the earliest low-cost slot that avoids
receiver contention. Expert placement, model colocation, and heterogeneous GPU
assignment are deliberately outside this fixed-placement reference.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.phase_local.common import (
    build_phase_serial_release_aware_plan,
    build_transfer_layouts_and_tasks,
    finalize_execution_plan,
    flows_from_matrix,
    include_real_p2_phase,
)


class AuroraOrderFixedPolicy:
    policy_name = "aurora_order_reference"
    policy_version = "v2-conflict-avoid"
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
        p0_waves, p0_trace = _arrange_logical_phase(
            flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
            start_wave_id=0,
        )
        p1_waves, p1_trace = _arrange_logical_phase(
            flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True),
            start_wave_id=len(p0_waves),
        )
        p2_waves: list[LogicalWave] = []
        p2_trace: list[dict[str, Any]] = []
        if include_real_p2_phase(problem):
            p2_waves, p2_trace = _arrange_logical_phase(
                flows_from_matrix(
                    problem.p2_next_dispatch_forecast_matrix,
                    phase="p2_next_dispatch",
                    release_state="ready",
                    executable=True,
                ),
                start_wave_id=len(p0_waves) + len(p1_waves),
            )
        trace = p0_trace + p1_trace + p2_trace
        base = build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="phase_local_aurora_fixed_placement",
            tie_break_rule="busiest sender first; minimum slot-extension conflict-free placement",
            priority_components={
                "source_order": "descending outgoing traffic",
                "slot_rule": "avoid receiver conflict then minimize projected wave duration",
                "placement": "fixed",
            },
            p0_waves=tuple(p0_waves),
            p1_waves=tuple(p1_waves),
            p2_waves=tuple(p2_waves),
            service_model="phase_serial_aurora_order_v2",
        )
        return LogicalSchedulePlan(
            policy_name=base.policy_name,
            waves=base.waves,
            diagnostics={
                **base.diagnostics,
                "literature_mapping": "style",
                "implemented_core": (
                    "bottleneck_sender_seed",
                    "descending_sender_load_order",
                    "receiver_contention_avoiding_transmission_order",
                ),
                "missing_mechanisms": (
                    "expert_colocation",
                    "heterogeneous_gpu_assignment",
                    "multi_model_overlap",
                ),
                "aurora_trace": trace,
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
        started_ns = time.perf_counter_ns()
        wave_tasks, trace = _arrange_items(all_tasks)
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
            "priority_components": {
                "source_order": "descending outgoing traffic",
                "slot_rule": "avoid receiver conflict then minimize projected wave duration",
                "selection_trace": trace,
            },
            "tie_break_rule": "projected duration increase, wave duration, wave id, destination",
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


def _item_id(item: Any) -> str:
    return str(getattr(item, "task_id", getattr(item, "flow_id", "")))


def _item_bytes(item: Any) -> int:
    return int(getattr(item, "byte_count"))


def _arrange_items(items: list[Any] | tuple[Any, ...]) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    pending = list(items)
    if not pending:
        return [], []
    source_total: dict[int, int] = defaultdict(int)
    destination_total: dict[int, int] = defaultdict(int)
    by_source: dict[int, list[Any]] = defaultdict(list)
    for item in pending:
        src = int(item.src_rank)
        dst = int(item.dst_rank)
        size = _item_bytes(item)
        source_total[src] += size
        destination_total[dst] += size
        by_source[src].append(item)
    bottleneck_gpu = min(
        set(source_total) | set(destination_total),
        key=lambda rank: (-max(source_total.get(rank, 0), destination_total.get(rank, 0)), rank),
    )
    bottleneck_sender = min(source_total, key=lambda rank: (-source_total[rank], rank))
    source_order = [bottleneck_sender] + [
        src for src in sorted(source_total, key=lambda rank: (-source_total[rank], rank)) if src != bottleneck_sender
    ]
    waves: list[list[Any]] = []
    wave_sources: list[set[int]] = []
    wave_destinations: list[set[int]] = []
    wave_duration: list[int] = []
    trace: list[dict[str, Any]] = []
    for source_position, src in enumerate(source_order):
        source_items = sorted(
            by_source[src],
            key=lambda item: (-_item_bytes(item), int(item.dst_rank), _item_id(item)),
        )
        for item in source_items:
            dst = int(item.dst_rank)
            size = _item_bytes(item)
            candidates: list[tuple[tuple[int, int, int, int], int]] = []
            for wave_id in range(len(waves)):
                if src in wave_sources[wave_id] or dst in wave_destinations[wave_id]:
                    continue
                projected = max(wave_duration[wave_id], size)
                candidates.append(
                    (
                        (
                            projected - wave_duration[wave_id],
                            projected,
                            wave_id,
                            dst,
                        ),
                        wave_id,
                    )
                )
            if candidates:
                _, selected_wave = min(candidates)
                reason = "existing_conflict_free_slot"
            else:
                selected_wave = len(waves)
                waves.append([])
                wave_sources.append(set())
                wave_destinations.append(set())
                wave_duration.append(0)
                reason = "new_slot"
            before_duration = wave_duration[selected_wave]
            waves[selected_wave].append(item)
            wave_sources[selected_wave].add(src)
            wave_destinations[selected_wave].add(dst)
            wave_duration[selected_wave] = max(wave_duration[selected_wave], size)
            trace.append(
                {
                    "source_position": source_position,
                    "bottleneck_gpu": bottleneck_gpu,
                    "bottleneck_sender": bottleneck_sender,
                    "src_rank": src,
                    "dst_rank": dst,
                    "item_id": _item_id(item),
                    "byte_count": size,
                    "selected_wave": selected_wave,
                    "wave_duration_before": before_duration,
                    "wave_duration_after": wave_duration[selected_wave],
                    "selection_reason": reason,
                }
            )
    return waves, trace


def _arrange_logical_phase(
    flows: tuple[FlowDemand, ...],
    *,
    start_wave_id: int,
) -> tuple[list[LogicalWave], list[dict[str, Any]]]:
    wave_flows, trace = _arrange_items(flows)
    waves = [
        LogicalWave(
            wave_id=start_wave_id + wave_offset,
            flows=tuple(items),
            duration=float(max((_item_bytes(item) for item in items), default=0)),
        )
        for wave_offset, items in enumerate(wave_flows)
    ]
    phase = flows[0].phase if flows else ""
    for row in trace:
        row["phase"] = phase
        row["selected_wave"] = int(start_wave_id + int(row["selected_wave"]))
    return waves, trace


__all__ = ["AuroraOrderFixedPolicy"]
