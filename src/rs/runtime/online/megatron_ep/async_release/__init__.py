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
from .agreement import build_async_release_order_digest, gather_and_validate_async_release_schedule, validate_async_release_global_agreement
from .compiled_schedule import CompiledAsyncReleaseSchedule, compile_async_release_schedule, decode_compiled_async_release_schedule
from .executor import AsyncReleaseExecutor, AsyncReleaseExecutorConfig
from .joint_plan_agreement import (
    GlobalJointPlanWire,
    agree_global_joint_plan,
    validate_local_schedule_against_global_plan,
    validate_pairwise_send_recv_contracts,
)
from .plan_builder import AsyncReleasePlanBuilder, validate_async_release_execution_plan
from .p2p_executor import AsyncReleaseP2PExecutor, AsyncReleaseP2PExecutorConfig, AsyncReleaseRankContext
from .runtime_plan_builder import AsyncReleaseRuntimePlanBuilder, build_runtime_async_release_task_id
from .runtime_projection import (
    HostProjectedPlan,
    RuntimeHostFeasibilityProjector,
    host_project_safe_selection,
)
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
    "GlobalJointPlanWire",
    "HostProjectedPlan",
    "AsyncReleaseP2PExecutor",
    "AsyncReleaseP2PExecutorConfig",
    "AsyncReleaseRankContext",
    "AsyncReleaseRuntimePlanBuilder",
    "RuntimeHostFeasibilityProjector",
    "agree_global_joint_plan",
    "apply_event",
    "blocked_task_ids",
    "build_async_release_order_digest",
    "build_shadow_plan_from_matrices",
    "compile_async_release_schedule",
    "decode_compiled_async_release_schedule",
    "decide_next_action",
    "gather_and_validate_async_release_schedule",
    "host_project_safe_selection",
    "mark_task_completed",
    "mark_task_released",
    "ready_task_ids",
    "register_shadow_plan",
    "simulate_async_release",
    "build_runtime_async_release_task_id",
    "validate_async_release_global_agreement",
    "validate_async_release_execution_plan",
    "validate_local_schedule_against_global_plan",
    "validate_pairwise_send_recv_contracts",
    "validate_async_release_state",
    "validate_shadow_plan",
]
