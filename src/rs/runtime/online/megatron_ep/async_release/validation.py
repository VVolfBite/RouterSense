"""Validation helpers for the async-release shadow-only skeleton."""

from __future__ import annotations

from typing import Any

from .contracts import AsyncReleaseDecision, AsyncShadowPlan
from .state import AsyncReleaseState


def validate_shadow_plan(plan: AsyncShadowPlan) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_task_ids: set[str] = set()
    for task in plan.tasks:
        if task.task_id in seen_task_ids:
            errors.append(f"duplicate task_id: {task.task_id}")
        seen_task_ids.add(task.task_id)
        if task.dependency != "forecast_only" and int(task.byte_count) <= 0:
            errors.append(f"non-forecast task must have positive byte_count: {task.task_id}")
        if task.dependency == "forecast_only" and plan.is_executable:
            errors.append(f"forecast_only task cannot belong to executable shadow plan: {task.task_id}")
    if not plan.is_executable and not str(plan.reason_not_executable):
        errors.append("non-executable shadow plan must provide reason_not_executable")
    if plan.is_executable:
        warnings.append("async_release skeleton currently should not emit executable shadow plans")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def validate_async_release_state(
    state: AsyncReleaseState,
    decision: AsyncReleaseDecision | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_task_ids: set[str] = set()
    for task_id, task in state.tasks_by_id.items():
        if task_id in seen_task_ids:
            errors.append(f"duplicate task_id in state: {task_id}")
        seen_task_ids.add(task_id)
        if task.dependency != "forecast_only" and int(task.byte_count) <= 0:
            errors.append(f"non-forecast task must have positive byte_count: {task_id}")
        if task.release_state == "completed" and task.dependency == "forecast_only":
            errors.append(f"forecast_only task cannot complete: {task_id}")
        if task.release_state == "completed" and task.phase == "P1":
            completed_p0_ranks = set(state.completed_ranks_by_phase.get("P0", ()))
            if int(task.src_rank) not in completed_p0_ranks:
                errors.append(f"P1 task completed before P0 completion: {task_id}")
        if task.release_state == "released" and task.dependency == "forecast_only":
            errors.append(f"forecast_only task cannot be released: {task_id}")
    for plan in state.shadow_plans:
        result = validate_shadow_plan(plan)
        errors.extend(result["errors"])
        warnings.extend(result["warnings"])
    if state.fallback_required and decision is not None and decision.action == "release_ready_tasks":
        errors.append("fallback_required state must not continue releasing ready tasks")
    return {"valid": not errors, "errors": errors, "warnings": warnings}
