"""State-only async-release shadow runtime model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .contracts import AsyncReleaseEvent, AsyncReleaseTask, AsyncReleaseWindowKey, AsyncShadowPlan


@dataclass(frozen=True)
class AsyncReleaseState:
    window_key: AsyncReleaseWindowKey
    seen_events: tuple[AsyncReleaseEvent, ...] = ()
    tasks_by_id: dict[str, AsyncReleaseTask] | None = None
    shadow_plans: tuple[AsyncShadowPlan, ...] = ()
    completed_ranks_by_phase: dict[str, tuple[int, ...]] | None = None
    materialized_ranks_by_phase: dict[str, tuple[int, ...]] | None = None
    fallback_required: bool = False

    def __post_init__(self) -> None:
        if self.tasks_by_id is None:
            object.__setattr__(self, "tasks_by_id", {})
        if self.completed_ranks_by_phase is None:
            object.__setattr__(self, "completed_ranks_by_phase", {})
        if self.materialized_ranks_by_phase is None:
            object.__setattr__(self, "materialized_ranks_by_phase", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_key": self.window_key.to_dict(),
            "seen_events": [event.to_dict() for event in self.seen_events],
            "tasks_by_id": {task_id: task.to_dict() for task_id, task in self.tasks_by_id.items()},
            "shadow_plans": [plan.to_dict() for plan in self.shadow_plans],
            "completed_ranks_by_phase": {phase: list(ranks) for phase, ranks in self.completed_ranks_by_phase.items()},
            "materialized_ranks_by_phase": {
                phase: list(ranks) for phase, ranks in self.materialized_ranks_by_phase.items()
            },
            "fallback_required": self.fallback_required,
        }


def _sorted_rank_tuple(values: set[int]) -> tuple[int, ...]:
    return tuple(sorted(values))


def apply_event(state: AsyncReleaseState, event: AsyncReleaseEvent) -> AsyncReleaseState:
    completed = {phase: set(ranks) for phase, ranks in state.completed_ranks_by_phase.items()}
    materialized = {phase: set(ranks) for phase, ranks in state.materialized_ranks_by_phase.items()}
    fallback_required = state.fallback_required or event.event_type == "fallback_required"
    if event.event_type.endswith("_completed"):
        completed.setdefault(event.phase, set()).add(int(event.rank))
    if event.event_type == "p1_materialized":
        materialized.setdefault(event.phase, set()).add(int(event.rank))
    return replace(
        state,
        seen_events=state.seen_events + (event,),
        completed_ranks_by_phase={phase: _sorted_rank_tuple(ranks) for phase, ranks in completed.items()},
        materialized_ranks_by_phase={phase: _sorted_rank_tuple(ranks) for phase, ranks in materialized.items()},
        fallback_required=fallback_required,
    )


def register_shadow_plan(state: AsyncReleaseState, plan: AsyncShadowPlan) -> AsyncReleaseState:
    tasks = dict(state.tasks_by_id)
    for task in plan.tasks:
        tasks.setdefault(task.task_id, task)
    return replace(
        state,
        tasks_by_id=tasks,
        shadow_plans=state.shadow_plans + (plan,),
    )


def mark_task_released(state: AsyncReleaseState, task_id: str) -> AsyncReleaseState:
    task = state.tasks_by_id.get(task_id)
    if task is None or task.release_state == "completed":
        return state
    if task.release_state not in {"ready", "blocked"}:
        return state
    tasks = dict(state.tasks_by_id)
    tasks[task_id] = replace(task, release_state="released")
    return replace(state, tasks_by_id=tasks)


def mark_task_completed(state: AsyncReleaseState, task_id: str) -> AsyncReleaseState:
    task = state.tasks_by_id.get(task_id)
    if task is None or task.release_state != "released":
        return state
    tasks = dict(state.tasks_by_id)
    tasks[task_id] = replace(task, release_state="completed")
    completed = {phase: set(ranks) for phase, ranks in state.completed_ranks_by_phase.items()}
    completed.setdefault(task.phase, set()).add(int(task.src_rank))
    return replace(
        state,
        tasks_by_id=tasks,
        completed_ranks_by_phase={phase: _sorted_rank_tuple(ranks) for phase, ranks in completed.items()},
    )


def ready_task_ids(state: AsyncReleaseState) -> tuple[str, ...]:
    return tuple(
        sorted(
            task_id
            for task_id, task in state.tasks_by_id.items()
            if task.release_state in {"ready", "released"} and task.dependency != "forecast_only"
        )
    )


def blocked_task_ids(state: AsyncReleaseState) -> tuple[str, ...]:
    return tuple(sorted(task_id for task_id, task in state.tasks_by_id.items() if task.release_state == "blocked"))
