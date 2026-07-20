"""Shadow-only async-release planner skeleton.

这里只表达未来 async-release runtime 需要维护的 release / forecast 语义，
不生成真实 online executor 可直接下发的通信命令。
"""

from __future__ import annotations

from typing import Iterable

from rs.scheduling.validation import stable_hash

from .contracts import (
    AsyncReleaseDecision,
    AsyncReleaseTask,
    AsyncReleaseWindowKey,
    AsyncShadowPlan,
)
from .state import AsyncReleaseState, ready_task_ids


def _iter_tasks(
    matrix: tuple[tuple[int, ...], ...],
    *,
    phase: str,
    release_state: str,
    dependency: str,
    source: str,
) -> Iterable[AsyncReleaseTask]:
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            yield AsyncReleaseTask(
                task_id=f"{phase}:{src_rank}->{dst_rank}",
                phase=phase,
                src_rank=src_rank,
                dst_rank=dst_rank,
                byte_count=int(byte_count),
                release_state=release_state,
                dependency=dependency,
                source=source,
            )


def build_shadow_plan_from_matrices(
    *,
    window_key: AsyncReleaseWindowKey,
    policy_name: str,
    created_at_event: str,
    applies_to_layer_id: str,
    p0_dispatch_matrix: tuple[tuple[int, ...], ...],
    p1_return_matrix: tuple[tuple[int, ...], ...],
    p2_next_dispatch_forecast_matrix: tuple[tuple[int, ...], ...],
    forecast_digest: str = "",
) -> AsyncShadowPlan:
    tasks = tuple(
        list(
            _iter_tasks(
                p0_dispatch_matrix,
                phase="P0",
                release_state="ready",
                dependency="none",
                source="actual",
            )
        )
        + list(
            _iter_tasks(
                p1_return_matrix,
                phase="P1",
                release_state="blocked",
                dependency="wait_p0_complete",
                source="actual",
            )
        )
        + list(
            _iter_tasks(
                p2_next_dispatch_forecast_matrix,
                phase="P2",
                release_state="blocked",
                dependency="forecast_only",
                source="predicted",
            )
        )
    )
    has_forecast = any(task.phase == "P2" for task in tasks)
    plan_id = stable_hash(
        {
            "window_key": window_key.to_dict(),
            "policy_name": policy_name,
            "applies_to_layer_id": applies_to_layer_id,
            "tasks": [task.to_dict() for task in tasks],
            "forecast_digest": forecast_digest,
        }
    )
    return AsyncShadowPlan(
        plan_id=f"async-shadow:{plan_id}",
        window_key=window_key,
        policy_name=policy_name,
        created_at_event=created_at_event,
        applies_to_layer_id=applies_to_layer_id,
        tasks=tasks,
        forecast_digest=forecast_digest,
        is_executable=False,
        reason_not_executable="shadow_only_runtime_contract" if has_forecast else "no_forecast_tasks_available",
    )


def decide_next_action(state: AsyncReleaseState) -> AsyncReleaseDecision:
    ready_ids = ready_task_ids(state)
    if state.fallback_required:
        return AsyncReleaseDecision(
            decision_id=f"decision:{stable_hash({'action': 'fallback', 'events': len(state.seen_events)})}",
            window_key=state.window_key,
            rank=-1,
            action="fallback_phase_sync",
            task_ids=(),
            reason="fallback_required",
        )
    if ready_ids:
        return AsyncReleaseDecision(
            decision_id=f"decision:{stable_hash({'action': 'release', 'task_ids': ready_ids})}",
            window_key=state.window_key,
            rank=-1,
            action="release_ready_tasks",
            task_ids=ready_ids,
            reason="ready_p0_tasks_available",
        )
    has_forecast = any(task.dependency == "forecast_only" for task in state.tasks_by_id.values())
    if has_forecast:
        return AsyncReleaseDecision(
            decision_id=f"decision:{stable_hash({'action': 'shadow', 'plans': len(state.shadow_plans)})}",
            window_key=state.window_key,
            rank=-1,
            action="prepare_shadow_plan",
            task_ids=(),
            reason="forecast_pressure_available",
        )
    return AsyncReleaseDecision(
        decision_id=f"decision:{stable_hash({'action': 'hold', 'events': len(state.seen_events)})}",
        window_key=state.window_key,
        rank=-1,
        action="hold",
        task_ids=(),
        reason="no_ready_or_forecast_tasks",
    )
