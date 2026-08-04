from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Any

from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext, PlanWave

from .fifo import build_transfer_layouts_and_tasks, finalize_execution_plan
from ..capabilities import PolicyCapabilities


def _hint_rank_pressure(hint_digest: str, rank: int) -> int:
    blob = hashlib.sha256(f"{hint_digest}:{rank}".encode("utf-8")).hexdigest()[:8]
    return int(blob, 16) % 1000


def _preferred_edge_priority(metadata: dict[str, Any], *, phase: str) -> dict[tuple[int, int], int]:
    priority: dict[tuple[int, int], int] = {}
    for item in metadata.get("preferred_edges", ()) or ():
        if str(item.get("phase", "")) != str(phase):
            continue
        key = (int(item.get("src_rank", -1)), int(item.get("dst_rank", -1)))
        current = int(item.get("priority", len(priority)))
        if key not in priority or current < priority[key]:
            priority[key] = current
    return priority


class RouterSenseP0P1P2HintPolicy:
    policy_name = "routersense_p0p1p2_hint"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=True,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=True,
        uses_p2_forecast=True,
        requires_fixed_placement=False,
        evaluation_eligible=False,
    )

    def __init__(self, *, bucket_rows: int, p0_weight: float = 1.0, p1_reservation_weight: float = 1.0, p2_hint_weight: float = 1.0) -> None:
        self.bucket_rows = int(bucket_rows)
        self.p0_weight = float(p0_weight)
        self.p1_reservation_weight = float(p1_reservation_weight)
        self.p2_hint_weight = float(p2_hint_weight)

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
        future_out_pressure = defaultdict(int)
        future_in_pressure = defaultdict(int)
        for layout in transfer_layouts:
            if int(layout.src_rank) == int(layout.dst_rank):
                continue
            future_out_pressure[int(layout.dst_rank)] += int(layout.byte_count)
            future_in_pressure[int(layout.src_rank)] += int(layout.byte_count)
        hint_available = local_context.p2_hint.hint_mode != "none"
        hint_pressure = {
            int(rank): (
                _hint_rank_pressure(local_context.p2_hint.hint_digest, int(rank))
                if hint_available and self.p2_hint_weight > 0.0
                else 0
            )
            for rank in local_context.ep_group_ranks
        }
        edge_priority = (
            _preferred_edge_priority(local_context.p2_hint.metadata, phase=local_context.phase)
            if hint_available
            else {}
        )
        use_prepared_priority = hint_available and bool(edge_priority) and self.p2_hint_weight > 0.0
        matched_hint_edges = {
            (int(task.src_rank), int(task.dst_rank))
            for task in all_tasks
            if (int(task.src_rank), int(task.dst_rank)) in edge_priority
        }

        def task_priority(task) -> tuple:
            edge_key = (int(task.src_rank), int(task.dst_rank))
            plan_priority = edge_priority.get(edge_key)
            base_score = (
                self.p0_weight * float(task.byte_count)
                + self.p1_reservation_weight * float(future_out_pressure[int(task.dst_rank)] + future_in_pressure[int(task.src_rank)])
                + self.p2_hint_weight * float(hint_pressure[int(task.dst_rank)] + hint_pressure[int(task.src_rank)])
            )
            if use_prepared_priority:
                return (
                    0 if plan_priority is not None else 1,
                    int(plan_priority) if plan_priority is not None else 10**9,
                    -base_score,
                    -int(task.byte_count),
                    int(task.src_rank),
                    int(task.dst_rank),
                    int(task.bucket_ordinal),
                )
            return (
                -base_score,
                -int(task.byte_count),
                int(task.src_rank),
                int(task.dst_rank),
                int(task.bucket_ordinal),
            )

        sort_started_ns = time.perf_counter_ns()
        ordered_tasks = sorted(
            all_tasks,
            key=task_priority,
        )
        sort_time_us = (time.perf_counter_ns() - sort_started_ns) / 1000.0
        waves: list[PlanWave] = []
        pending = ordered_tasks[:]
        wave_id = 0
        pack_started_ns = time.perf_counter_ns()
        while pending:
            used_outgoing, used_incoming, chosen, remaining = set(), set(), [], []
            for task in pending:
                if int(task.src_rank) in used_outgoing or int(task.dst_rank) in used_incoming:
                    remaining.append(task)
                    continue
                chosen.append(task)
                used_outgoing.add(int(task.src_rank))
                used_incoming.add(int(task.dst_rank))
            waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(chosen)))
            pending = remaining
            wave_id += 1
        pack_time_us = (time.perf_counter_ns() - pack_started_ns) / 1000.0
        p2_influenced_plan = hint_available and (bool(edge_priority) or self.p2_hint_weight > 0.0)
        evaluation_eligible = local_context.p2_hint.hint_mode == "calibrated_artifact"
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_capabilities": self.capabilities.to_dict(),
            "bucket_order": [task.task_id for task in ordered_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": [float(sum(int(task.byte_count) for task in wave.bucket_tasks)) for wave in waves],
            "uses_current_phase_demand": True,
            "uses_p1_reservation": True,
            "uses_p2_hint": True,
            "priority_components": {
                "p0_weight": self.p0_weight,
                "p1_reservation_weight": self.p1_reservation_weight,
                "p2_hint_weight": self.p2_hint_weight,
                "p2_hint_pressure": {str(k): int(v) for k, v in hint_pressure.items()},
                "prepared_edge_priority": {
                    f"{src}->{dst}": int(priority)
                    for (src, dst), priority in sorted(edge_priority.items(), key=lambda item: item[1])
                },
            },
            "tie_break_rule": "src_rank,dst_rank,bucket_ordinal",
            "fallback_reason": "",
            "evaluation_eligible": evaluation_eligible,
            "p2_hint_source": local_context.p2_hint.hint_source,
            "p2_hint_digest": local_context.p2_hint.hint_digest,
            "p2_hint_mode": local_context.p2_hint.hint_mode,
            "p2_hint_available": hint_available,
            "p2_influenced_plan": p2_influenced_plan,
            "prepared_plan_hint_available": hint_available and bool(edge_priority),
            "ordered_by_prepared_plan": bool(use_prepared_priority),
            "hint_edges_available": len(edge_priority),
            "hint_edges_matched": len(matched_hint_edges),
            "hint_edges_consumed": len(matched_hint_edges),
            "hint_match_rate": float(len(matched_hint_edges) / len(edge_priority)) if edge_priority else 0.0,
            "preferred_edge_count": int(local_context.p2_hint.metadata.get("preferred_edge_count", len(edge_priority)) or 0),
            "preferred_wave_count": int(local_context.p2_hint.metadata.get("preferred_wave_count", 0) or 0),
            "forecast_consumed": bool(p2_influenced_plan),
            "prediction_used": bool(p2_influenced_plan),
            "source_logical_plan_hash": str(local_context.p2_hint.metadata.get("source_logical_plan_hash", "")),
            "prepared_window_key": str(local_context.p2_hint.metadata.get("window_key", "")),
        }
        return finalize_execution_plan(
            local_context=local_context,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            bucket_rows=self.bucket_rows,
            transfer_layouts=transfer_layouts,
            all_tasks=ordered_tasks,
            waves=tuple(waves),
            diagnostics=diagnostics,
            timing_metrics={
                **build_stats,
                "sort_tasks_time_us": sort_time_us,
                "pack_phase_tasks_time_us": pack_time_us,
                "wave_count": int(len(waves)),
                "max_wave_task_count": int(max((len(wave.bucket_tasks) for wave in waves), default=0)),
                "task_count": int(len(ordered_tasks)),
            },
        )
