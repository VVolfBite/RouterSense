"""Async-release shadow-only contracts and state skeleton."""

from .contracts import (
    AsyncReleaseExecutionPlan,
    AsyncReleasePreparedPlan,
    AsyncReleaseDecision,
    AsyncReleaseEvent,
    AsyncReleaseTask,
    AsyncReleaseWindowKey,
    AsyncShadowPlan,
)
from .agreement import build_async_release_order_digest, validate_async_release_global_agreement
from .compiled_schedule import CompiledAsyncReleaseSchedule, compile_async_release_schedule, decode_compiled_async_release_schedule
from .executor import AsyncReleaseExecutor, AsyncReleaseExecutorConfig
from .plan_builder import AsyncReleasePlanBuilder, validate_async_release_execution_plan
from .shadow_controller import build_shadow_plan_from_matrices, decide_next_action
from .simulator import simulate_async_release
from .state import (
    AsyncReleaseState,
    apply_event,
    blocked_task_ids,
    mark_task_completed,
    mark_task_released,
    ready_task_ids,
    register_shadow_plan,
)
from .validation import validate_async_release_state, validate_shadow_plan

__all__ = [
    "AsyncReleaseDecision",
    "AsyncReleaseExecutor",
    "AsyncReleaseExecutorConfig",
    "AsyncReleaseExecutionPlan",
    "AsyncReleaseEvent",
    "AsyncReleasePreparedPlan",
    "AsyncReleasePlanBuilder",
    "AsyncReleaseTask",
    "AsyncReleaseWindowKey",
    "AsyncReleaseState",
    "AsyncShadowPlan",
    "CompiledAsyncReleaseSchedule",
    "apply_event",
    "blocked_task_ids",
    "build_async_release_order_digest",
    "build_shadow_plan_from_matrices",
    "compile_async_release_schedule",
    "decode_compiled_async_release_schedule",
    "decide_next_action",
    "mark_task_completed",
    "mark_task_released",
    "ready_task_ids",
    "register_shadow_plan",
    "simulate_async_release",
    "validate_async_release_global_agreement",
    "validate_async_release_execution_plan",
    "validate_async_release_state",
    "validate_shadow_plan",
]
