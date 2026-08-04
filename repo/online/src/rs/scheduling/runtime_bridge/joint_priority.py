from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from rs.scheduling.contracts import LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext, PlanWave
from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_diagonal_report,
    matrix_row_sums_remote,
    matrix_col_sums_remote,
)

from ..capabilities import PolicyCapabilities
from ..phase_local.common import (
    build_phase_serial_release_aware_plan,
    build_transfer_layouts_and_tasks,
    finalize_execution_plan,
    flows_from_matrix,
    include_real_p2_phase,
)


def _row_sums(matrix: tuple[tuple[int, ...], ...]) -> list[int]:
    return [int(value) for value in matrix_row_sums_remote(matrix)]


def _col_sums(matrix: tuple[tuple[int, ...], ...]) -> list[int]:
    return [int(value) for value in matrix_col_sums_remote(matrix)]


def _joint_rank_pressure(
    *,
    p1_matrix: tuple[tuple[int, ...], ...],
    p2_matrix: tuple[tuple[int, ...], ...],
) -> dict[int, int]:
    p1_rows = _row_sums(p1_matrix)
    p1_cols = _col_sums(p1_matrix)
    p2_rows = _row_sums(p2_matrix)
    p2_cols = _col_sums(p2_matrix)
    rank_pressure: dict[int, int] = {}
    for rank in range(max(len(p1_rows), len(p2_rows), len(p1_cols), len(p2_cols), 0)):
        rank_pressure[rank] = int(
            (p1_rows[rank] if rank < len(p1_rows) else 0)
            + (p1_cols[rank] if rank < len(p1_cols) else 0)
            + (p2_rows[rank] if rank < len(p2_rows) else 0)
            + (p2_cols[rank] if rank < len(p2_cols) else 0)
        )
    return rank_pressure


def _priority_metadata(
    *,
    p1_matrix: tuple[tuple[int, ...], ...],
    p2_matrix: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    p1_diag = matrix_diagonal_report(p1_matrix)
    p2_diag = matrix_diagonal_report(p2_matrix)
    return {
        "p1_row_sums": _row_sums(p1_matrix),
        "p1_col_sums": _col_sums(p1_matrix),
        "p2_row_sums": _row_sums(p2_matrix),
        "p2_col_sums": _col_sums(p2_matrix),
        "joint_rank_pressure": _joint_rank_pressure(p1_matrix=p1_matrix, p2_matrix=p2_matrix),
        "self_bytes_ignored": int(p1_diag["self_bytes"]) + int(p2_diag["self_bytes"]),
    }


class RouterSenseJointPriorityPhaseSyncPolicy:
    policy_name = "routersense_joint_priority_phase_sync"
    policy_version = "v1"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=True,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=True,
        uses_p2_forecast=True,
        requires_fixed_placement=False,
        evaluation_eligible=True,
    )

    def __init__(self, *, bucket_rows: int, p0_weight: float = 1.0, p1_reservation_weight: float = 1.0, p2_hint_weight: float = 1.0) -> None:
        self.bucket_rows = int(bucket_rows)
        self.p0_weight = float(p0_weight)
        self.p1_reservation_weight = float(p1_reservation_weight)
        self.p2_hint_weight = float(p2_hint_weight)

    def _flow_priority(
        self,
        *,
        phase: str,
        src_rank: int,
        dst_rank: int,
        byte_count: int,
        meta: dict[str, Any],
    ) -> tuple[float, float, int, int]:
        rank_pressure = meta["joint_rank_pressure"]
        p1_rows = meta["p1_row_sums"]
        p1_cols = meta["p1_col_sums"]
        p2_rows = meta["p2_row_sums"]
        p2_cols = meta["p2_col_sums"]
        if phase == "p0_dispatch":
            release_pressure = int((p1_rows[dst_rank] if dst_rank < len(p1_rows) else 0) + (p1_cols[dst_rank] if dst_rank < len(p1_cols) else 0))
            downstream_pressure = int((p2_rows[src_rank] if src_rank < len(p2_rows) else 0) + (p2_cols[src_rank] if src_rank < len(p2_cols) else 0))
            score = (
                self.p0_weight * float(byte_count)
                + self.p1_reservation_weight * float(release_pressure)
                + self.p2_hint_weight * float(downstream_pressure)
            )
            return (score, float(release_pressure), src_rank, dst_rank)
        downstream_pressure = int((p2_rows[dst_rank] if dst_rank < len(p2_rows) else 0) + (p2_cols[dst_rank] if dst_rank < len(p2_cols) else 0))
        score = self.p0_weight * float(byte_count) + self.p2_hint_weight * float(downstream_pressure)
        return (score, float(rank_pressure.get(src_rank, 0)), src_rank, dst_rank)

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        p1_matrix = canonicalize_remote_matrix(problem.p1_return_matrix)
        p2_matrix = canonicalize_remote_matrix(problem.p2_next_dispatch_forecast_matrix)
        meta = _priority_metadata(p1_matrix=p1_matrix, p2_matrix=p2_matrix)
        ordered_p0 = sorted(
            flows_from_matrix(problem.p0_dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
            key=lambda flow: (
                -self._flow_priority(
                    phase="p0_dispatch",
                    src_rank=int(flow.src_rank),
                    dst_rank=int(flow.dst_rank),
                    byte_count=int(flow.byte_count),
                    meta=meta,
                )[0],
                -self._flow_priority(
                    phase="p0_dispatch",
                    src_rank=int(flow.src_rank),
                    dst_rank=int(flow.dst_rank),
                    byte_count=int(flow.byte_count),
                    meta=meta,
                )[1],
                -int(flow.byte_count),
                int(flow.src_rank),
                int(flow.dst_rank),
            ),
        )
        ordered_p1 = sorted(
            flows_from_matrix(problem.p1_return_matrix, phase="p1_return", release_state="ready", executable=True),
            key=lambda flow: (
                -self._flow_priority(
                    phase="p1_return",
                    src_rank=int(flow.src_rank),
                    dst_rank=int(flow.dst_rank),
                    byte_count=int(flow.byte_count),
                    meta=meta,
                )[0],
                -int(flow.byte_count),
                int(flow.src_rank),
                int(flow.dst_rank),
            ),
        )
        ordered_p2 = []
        if include_real_p2_phase(problem):
            ordered_p2 = sorted(
                flows_from_matrix(problem.p2_next_dispatch_forecast_matrix, phase="p2_next_dispatch", release_state="ready", executable=True),
                key=lambda flow: (
                    -self._flow_priority(
                        phase="p1_return",
                        src_rank=int(flow.src_rank),
                        dst_rank=int(flow.dst_rank),
                        byte_count=int(flow.byte_count),
                        meta=meta,
                    )[0],
                    -int(flow.byte_count),
                    int(flow.src_rank),
                    int(flow.dst_rank),
                ),
            )
        p0_waves = self._pack_logical_phase(ordered_p0, start_wave_id=0)
        p1_waves = self._pack_logical_phase(ordered_p1, start_wave_id=len(p0_waves))
        p2_waves = self._pack_logical_phase(ordered_p2, start_wave_id=len(p0_waves) + len(p1_waves))
        return build_phase_serial_release_aware_plan(
            problem=problem,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            capabilities=self.capabilities,
            information_mode="joint_priority_phase_sync",
            tie_break_rule="joint_priority_score desc, byte_count desc, src_rank,dst_rank",
            priority_components={
                "p0_weight": self.p0_weight,
                "p1_reservation_weight": self.p1_reservation_weight,
                "p2_hint_weight": self.p2_hint_weight,
                "joint_rank_pressure": {str(k): int(v) for k, v in meta["joint_rank_pressure"].items()},
                "remote_only_matrix_invariant": True,
                "self_bytes_ignored": int(meta["self_bytes_ignored"]),
            },
            p0_waves=p0_waves,
            p1_waves=p1_waves,
            p2_waves=p2_waves,
            service_model="phase_sync_joint_priority",
        )

    @staticmethod
    def _pack_logical_phase(flows: list[Any], *, start_wave_id: int) -> tuple[LogicalWave, ...]:
        waves: list[LogicalWave] = []
        pending = list(flows)
        wave_id = int(start_wave_id)
        while pending:
            used_src: set[int] = set()
            used_dst: set[int] = set()
            chosen: list[Any] = []
            remaining: list[Any] = []
            for flow in pending:
                if int(flow.src_rank) in used_src or int(flow.dst_rank) in used_dst:
                    remaining.append(flow)
                    continue
                chosen.append(flow)
                used_src.add(int(flow.src_rank))
                used_dst.add(int(flow.dst_rank))
            duration = float(max((int(flow.byte_count) for flow in chosen), default=0))
            waves.append(LogicalWave(wave_id=wave_id, flows=tuple(chosen), duration=duration))
            pending = remaining
            wave_id += 1
        return tuple(waves)

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
        p1_rows: defaultdict[int, int] = defaultdict(int)
        p1_cols: defaultdict[int, int] = defaultdict(int)
        preferred_edges: dict[tuple[int, int], int] = {}
        metadata = local_context.p2_hint.metadata or {}
        prepared_priority_mode = str(metadata.get("prepared_priority_mode", "mapped_p2_tiebreak") or "mapped_p2_tiebreak")
        for item in metadata.get("preferred_edges", ()) or ():
            phase_name = str(item.get("phase", ""))
            if phase_name != str(local_context.phase):
                continue
            preferred_edges[(int(item.get("src_rank", -1)), int(item.get("dst_rank", -1)))] = int(item.get("priority", 0))
        predicted_row_sums = [int(value) for value in metadata.get("predicted_row_sums", ()) or ()]
        predicted_col_sums = [int(value) for value in metadata.get("predicted_col_sums", ()) or ()]
        uses_real_p1_reservation = bool(metadata.get("has_real_p1_reservation", False))
        if uses_real_p1_reservation:
            for row_idx, value in enumerate(metadata.get("p1_reservation_row_sums", ()) or ()):
                p1_rows[int(row_idx)] = int(value)
            for col_idx, value in enumerate(metadata.get("p1_reservation_col_sums", ()) or ()):
                p1_cols[int(col_idx)] = int(value)

        def predicted_pressure(rank: int) -> int:
            row = predicted_row_sums[rank] if rank < len(predicted_row_sums) else 0
            col = predicted_col_sums[rank] if rank < len(predicted_col_sums) else 0
            return int(row + col)

        def task_key(task: Any) -> tuple[Any, ...]:
            edge = (int(task.src_rank), int(task.dst_rank))
            plan_priority = preferred_edges.get(edge)
            if local_context.phase == "P0":
                release_pressure = int(p1_rows[int(task.dst_rank)] + p1_cols[int(task.dst_rank)])
                downstream_pressure = predicted_pressure(int(task.src_rank))
            else:
                release_pressure = int(p1_rows[int(task.src_rank)] + p1_cols[int(task.src_rank)])
                downstream_pressure = int(predicted_pressure(int(task.dst_rank)))
            score = (
                self.p0_weight * float(task.byte_count)
                + self.p1_reservation_weight * float(release_pressure)
                + self.p2_hint_weight * float(downstream_pressure)
            )
            prepared_hint_bonus = 0.0
            if plan_priority is not None:
                prepared_hint_bonus = float(max(0, len(preferred_edges) - int(plan_priority)))
            if prepared_priority_mode == "live_score_only":
                return (
                    -score,
                    -int(task.byte_count),
                    int(task.src_rank),
                    int(task.dst_rank),
                    int(task.bucket_ordinal),
                )
            if prepared_priority_mode == "mapped_p2_bounded_bonus":
                return (
                    -(score + 1e-3 * prepared_hint_bonus),
                    -int(task.byte_count),
                    int(task.src_rank),
                    int(task.dst_rank),
                    int(task.bucket_ordinal),
                )
            return (
                -score,
                0 if plan_priority is not None else 1,
                int(plan_priority) if plan_priority is not None else 10**9,
                -int(task.byte_count),
                int(task.src_rank),
                int(task.dst_rank),
                int(task.bucket_ordinal),
            )

        sort_started_ns = time.perf_counter_ns()
        ordered_tasks = sorted(all_tasks, key=task_key)
        sort_time_us = (time.perf_counter_ns() - sort_started_ns) / 1000.0
        waves: list[PlanWave] = []
        pending = ordered_tasks[:]
        wave_id = 0
        pack_started_ns = time.perf_counter_ns()
        while pending:
            used_src: set[int] = set()
            used_dst: set[int] = set()
            chosen: list[Any] = []
            remaining: list[Any] = []
            for task in pending:
                if int(task.src_rank) in used_src or int(task.dst_rank) in used_dst:
                    remaining.append(task)
                    continue
                chosen.append(task)
                used_src.add(int(task.src_rank))
                used_dst.add(int(task.dst_rank))
            waves.append(PlanWave(wave_id=wave_id, phase=local_context.phase, bucket_tasks=tuple(chosen)))
            pending = remaining
            wave_id += 1
        pack_time_us = (time.perf_counter_ns() - pack_started_ns) / 1000.0
        diagnostics = {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "bucket_order": [task.task_id for task in ordered_tasks],
            "wave_edges": [[{"src_rank": int(task.src_rank), "dst_rank": int(task.dst_rank), "bucket_id": task.task_id} for task in wave.bucket_tasks] for wave in waves],
            "per_wave_matching_weight": [float(sum(int(task.byte_count) for task in wave.bucket_tasks)) for wave in waves],
            "uses_current_phase_demand": True,
            "uses_p1_reservation": bool(uses_real_p1_reservation),
            "p1_reservation_source": "prepared_window_observation" if uses_real_p1_reservation else "unavailable",
            "p1_reservation_weight_effective": float(self.p1_reservation_weight if uses_real_p1_reservation else 0.0),
            "uses_p2_hint": True,
            "priority_components": {
                "p0_weight": self.p0_weight,
                "p1_reservation_weight": self.p1_reservation_weight,
                "p2_hint_weight": self.p2_hint_weight,
                "joint_rank_pressure": {str(rank): int(p1_rows[rank] + p1_cols[rank] + predicted_pressure(rank)) for rank in sorted(set(p1_rows) | set(p1_cols) | set(range(len(predicted_row_sums))))},
                "remote_only_matrix_invariant": True,
                "self_bytes_ignored": int(metadata.get("predicted_self_bytes", 0) or 0),
            },
            "tie_break_rule": f"{prepared_priority_mode} -> byte_count -> src_rank,dst_rank,bucket_ordinal",
            "fallback_reason": "",
            "evaluation_eligible": local_context.p2_hint.hint_mode == "calibrated_artifact",
            "p2_hint_source": local_context.p2_hint.hint_source,
            "p2_hint_digest": local_context.p2_hint.hint_digest,
            "p2_hint_mode": local_context.p2_hint.hint_mode,
            "forecast_consumed": local_context.p2_hint.hint_mode != "none",
            "prediction_used": local_context.p2_hint.hint_mode != "none",
            "online_eligible": True,
            "async_release_required": False,
            "predictor_name": str(metadata.get("predictor_name", "")),
            "prepared_plan_consumed_prediction_digest": str(metadata.get("prediction_digest", "")),
            "prepared_plan_p2_source": str(metadata.get("p2_matrix_source", local_context.p2_hint.hint_source)),
            "prepared_priority_mode": prepared_priority_mode,
            "mapped_p2_edge_count": int(metadata.get("mapped_p2_edge_count", len(preferred_edges)) or 0),
            "stale_p0_p1_edge_count_ignored": int(metadata.get("stale_p0_p1_edge_count_ignored", 0) or 0),
            "live_score_changed_order": True,
            "prepared_hint_changed_order": bool(preferred_edges),
            "remote_only_matrix_invariant": True,
            "self_bytes_ignored": int(metadata.get("predicted_self_bytes", 0) or 0),
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
