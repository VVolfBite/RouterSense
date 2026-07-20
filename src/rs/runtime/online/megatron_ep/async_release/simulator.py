"""CPU-only async-release simulator.

这个模拟器不接真实 GPU executor，只表达：
- P0 在当前层观测后可立即 release
- P1 需要等待对应 rank 的 P0 inbound completion + compute delay
- P2 只作为 shadow / forecast priority，不直接执行
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_col_sums_remote,
    matrix_diagonal_report,
    matrix_remote_bytes,
    matrix_row_sums_remote,
)

from rs.scheduling.online_adapters.plan_priority import PlanPriorityArtifact

Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SimTask:
    task_id: str
    phase: str
    src_rank: int
    dst_rank: int
    byte_count: int
    ready_time: float
    score: float


def _matrix(matrix: Any) -> Matrix:
    return canonicalize_remote_matrix(matrix)


def _rank_pressure(matrix: Matrix) -> dict[int, int]:
    rows = matrix_row_sums_remote(matrix)
    cols = matrix_col_sums_remote(matrix)
    return {rank: int((rows[rank] if rank < len(rows) else 0) + (cols[rank] if rank < len(cols) else 0)) for rank in range(max(len(rows), len(cols), 0))}


def _pack_ready_wave(tasks: list[SimTask]) -> list[SimTask]:
    used_src: set[int] = set()
    used_dst: set[int] = set()
    chosen: list[SimTask] = []
    for task in sorted(tasks, key=lambda item: (-item.score, -item.byte_count, item.src_rank, item.dst_rank, item.task_id)):
        if task.src_rank in used_src or task.dst_rank in used_dst:
            continue
        chosen.append(task)
        used_src.add(task.src_rank)
        used_dst.add(task.dst_rank)
    return chosen


def simulate_async_release(
    *,
    p0_dispatch_matrix: Any,
    p1_return_matrix: Any,
    predicted_p2_matrix: Any,
    compute_delay: float = 0.0,
    planning_time_us: float = 0.0,
    prediction_time_us: float = 0.0,
    control_delay_us: float = 0.0,
    prediction_lead_time_us: float = 0.0,
    policy_name: str = "current:p012:joint:event:rscf",
    priority_artifact: PlanPriorityArtifact | None = None,
) -> dict[str, Any]:
    raw_p0 = tuple(tuple(int(value) for value in row) for row in p0_dispatch_matrix)
    raw_p1 = tuple(tuple(int(value) for value in row) for row in p1_return_matrix)
    raw_p2 = tuple(tuple(int(value) for value in row) for row in predicted_p2_matrix)
    p0 = _matrix(raw_p0)
    p1 = _matrix(raw_p1)
    p2 = _matrix(raw_p2)
    p1_pressure = _rank_pressure(p1)
    p2_pressure = _rank_pressure(p2)
    priority_lookup: dict[tuple[str, int, int], tuple[float, int]] = {}
    if priority_artifact is not None:
        for entry in priority_artifact.priority_entries:
            priority_lookup[(str(entry.phase), int(entry.src_rank), int(entry.dst_rank))] = (float(entry.priority_score), int(entry.wave_id))
    p0_inbound_remaining: dict[int, int] = {}
    for src_rank, row in enumerate(p0):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            p0_inbound_remaining[dst_rank] = p0_inbound_remaining.get(dst_rank, 0) + int(byte_count)

    p0_tasks: list[SimTask] = []
    p1_waiting: list[tuple[str, int, int, int]] = []
    for src_rank, row in enumerate(p0):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            key = ("p0_dispatch", src_rank, dst_rank)
            artifact_score = priority_lookup.get(key, (0.0, 0))[0]
            score = float(max(artifact_score, float(byte_count + p1_pressure.get(dst_rank, 0) + p2_pressure.get(dst_rank, 0))))
            p0_tasks.append(SimTask(f"P0:{src_rank}->{dst_rank}", "P0", src_rank, dst_rank, int(byte_count), 0.0, score))
    for src_rank, row in enumerate(p1):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            p1_waiting.append((f"P1:{src_rank}->{dst_rank}", src_rank, dst_rank, int(byte_count)))

    time_now = float(control_delay_us)
    completed_p0_by_rank: dict[int, float] = {}
    done: set[str] = set()
    released_p1_ids: set[str] = set()
    waves: list[dict[str, Any]] = []
    blocked_task_count = len(p1_waiting)
    early_release_task_count = 0
    fallback_replan_count = 0
    dependency_violations = 0

    while True:
        ready_tasks = [task for task in p0_tasks if task.task_id not in done and task.ready_time <= time_now]
        for task_id, src_rank, dst_rank, byte_count in list(p1_waiting):
            ready_at = completed_p0_by_rank.get(src_rank)
            if ready_at is None or ready_at + float(compute_delay) > time_now:
                continue
            if task_id not in released_p1_ids:
                early_release_task_count += 1
                released_p1_ids.add(task_id)
            key = ("p1_return", src_rank, dst_rank)
            artifact_score = priority_lookup.get(key, (0.0, 0))[0]
            score = float(max(artifact_score, float(byte_count + p2_pressure.get(src_rank, 0) + p2_pressure.get(dst_rank, 0))))
            ready_tasks.append(SimTask(task_id, "P1", src_rank, dst_rank, byte_count, ready_at + float(compute_delay), score))
        ready_tasks = [task for task in ready_tasks if task.task_id not in done]
        if not ready_tasks:
            pending_times: list[float] = []
            for task in p0_tasks:
                if task.task_id not in done:
                    pending_times.append(task.ready_time)
            for task_id, src_rank, _dst_rank, _byte_count in p1_waiting:
                if task_id in done:
                    continue
                ready_at = completed_p0_by_rank.get(src_rank)
                if ready_at is not None:
                    pending_times.append(ready_at + float(compute_delay))
            if not pending_times:
                break
            next_time = min(value for value in pending_times if value > time_now)
            time_now = next_time
            continue
        wave = _pack_ready_wave(ready_tasks)
        if not wave:
            fallback_replan_count += 1
            break
        start = time_now
        end = start + max(float(task.byte_count) for task in wave)
        waves.append(
            {
                "wave_id": len(waves),
                "start": start,
                "end": end,
                "task_ids": [task.task_id for task in wave],
                "phases": [task.phase for task in wave],
            }
        )
        for task in wave:
            done.add(task.task_id)
            if task.phase == "P0":
                p0_inbound_remaining[task.dst_rank] = max(0, p0_inbound_remaining.get(task.dst_rank, 0) - int(task.byte_count))
                if p0_inbound_remaining.get(task.dst_rank, 0) == 0:
                    completed_p0_by_rank[task.dst_rank] = max(float(end), completed_p0_by_rank.get(task.dst_rank, 0.0))
            elif task.phase == "P1":
                required = completed_p0_by_rank.get(task.src_rank)
                if required is None or float(start) < float(required + float(compute_delay)):
                    dependency_violations += 1
        time_now = end

    p0_completion = max(completed_p0_by_rank.values(), default=0.0)
    first_need_time = min((completed_p0_by_rank.get(src_rank, 0.0) + float(compute_delay) for _task_id, src_rank, _dst, _bytes in p1_waiting), default=0.0)
    planning_total_us = float(planning_time_us) + float(prediction_time_us)
    hidden_planning_us = min(planning_total_us, max(0.0, float(prediction_lead_time_us)))
    exposed_planning_us = max(0.0, planning_total_us - hidden_planning_us)
    hidden_fraction = 1.0 if planning_total_us <= 0.0 else hidden_planning_us / planning_total_us
    p0_diag = matrix_diagonal_report(raw_p0)
    p1_diag = matrix_diagonal_report(raw_p1)
    p2_diag = matrix_diagonal_report(raw_p2)
    return {
        "policy_name": policy_name,
        "simulator_mode": "async_release_formal_joint",
        "completion_time": float(time_now),
        "task_release_timeline": waves,
        "shadow_plan_ready_time_us": float(max(0.0, planning_total_us)),
        "shadow_plan_ready_before_needed_us": float(max(0.0, first_need_time - planning_total_us)),
        "planning_time_us": planning_total_us,
        "prediction_time_us": float(prediction_time_us),
        "control_delay_us": float(control_delay_us),
        "prediction_lead_time_us": float(prediction_lead_time_us),
        "hidden_planning_fraction": float(hidden_fraction),
        "exposed_planning_time_us": float(exposed_planning_us),
        "fallback_replan_count": int(fallback_replan_count),
        "early_release_task_count": int(early_release_task_count),
        "blocked_task_count": int(blocked_task_count),
        "dependency_violations": int(dependency_violations),
        "predicted_p2_total_bytes": int(matrix_remote_bytes(p2)),
        "p0_self_bytes_ignored": int(p0_diag["self_bytes"]),
        "p1_self_bytes_ignored": int(p1_diag["self_bytes"]),
        "predicted_p2_self_bytes_ignored": int(p2_diag["self_bytes"]),
        "remote_only_matrix_invariant": True,
        "p0_inbound_completion_max": float(p0_completion),
        "online_eligible": False,
        "async_release_required": True,
        "evaluation_mode": "async_release_sim",
        "used_priority_artifact": priority_artifact is not None,
        "priority_artifact_digest": None if priority_artifact is None else priority_artifact.priority_digest,
        "source_policy": "" if priority_artifact is None else str(priority_artifact.source_policy),
        "selected_policy": "" if priority_artifact is None else str(priority_artifact.selected_policy),
        "fallback_to_local": False if priority_artifact is None else bool(priority_artifact.fallback_to_local),
    }
