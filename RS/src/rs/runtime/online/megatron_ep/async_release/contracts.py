"""Async-release shadow-only runtime contracts.

这些 dataclass 只表达未来 async-release runtime 语义，不接入真实 executor。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


AsyncEventType = Literal[
    "p0_observed",
    "p0_plan_ready",
    "p0_released",
    "p0_completed",
    "p1_materialized",
    "p1_plan_ready",
    "p1_released",
    "p1_completed",
    "p2_forecast_ready",
    "shadow_plan_ready",
    "fallback_required",
]
AsyncReleaseStateName = Literal["blocked", "ready", "released", "completed"]
AsyncDependency = Literal["none", "wait_p0_complete", "wait_p1_materialized", "forecast_only"]
AsyncTaskSource = Literal["actual", "predicted", "shadow"]
AsyncDecisionAction = Literal["hold", "release_ready_tasks", "prepare_shadow_plan", "fallback_phase_sync"]


@dataclass(frozen=True)
class AsyncReleaseExecutionPlan:
    plan_id: str
    source_safe_policy: str
    priority_artifact_digest: str
    phase_tasks: tuple[dict[str, Any], ...]
    dependency_edges: tuple[tuple[str, str], ...]
    release_conditions: dict[str, dict[str, Any]]
    fallback_to_phase_sync: bool
    online_executor_eligible: bool
    debug_replay_only: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsyncReleasePreparedPlan:
    plan_id: str
    compiled_schedule_digest: str
    compiled_schedule_task_count: int
    global_order_agreement_required: bool
    global_order_agreement_passed: bool
    dependency_validation_passed: bool
    fallback_to_phase_sync: bool
    fallback_reason: str
    ready_task_count: int
    blocked_task_count: int
    dependency_violation_count: int
    debug_replay_only: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsyncReleaseWindowKey:
    run_id_digest: str
    layer_id: str
    ep_group_hash: str
    forward_epoch: str
    microbatch_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsyncReleaseEvent:
    event_id: str
    window_key: AsyncReleaseWindowKey
    rank: int
    layer_id: str
    phase: str
    event_type: AsyncEventType

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "window_key": self.window_key.to_dict(),
            "rank": self.rank,
            "layer_id": self.layer_id,
            "phase": self.phase,
            "event_type": self.event_type,
        }


@dataclass(frozen=True)
class AsyncReleaseTask:
    task_id: str
    phase: str
    src_rank: int
    dst_rank: int
    byte_count: int
    release_state: AsyncReleaseStateName
    dependency: AsyncDependency
    source: AsyncTaskSource

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AsyncShadowPlan:
    plan_id: str
    window_key: AsyncReleaseWindowKey
    policy_name: str
    created_at_event: str
    applies_to_layer_id: str
    tasks: tuple[AsyncReleaseTask, ...]
    forecast_digest: str
    is_executable: bool
    reason_not_executable: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "window_key": self.window_key.to_dict(),
            "policy_name": self.policy_name,
            "created_at_event": self.created_at_event,
            "applies_to_layer_id": self.applies_to_layer_id,
            "tasks": [task.to_dict() for task in self.tasks],
            "forecast_digest": self.forecast_digest,
            "is_executable": self.is_executable,
            "reason_not_executable": self.reason_not_executable,
        }


@dataclass(frozen=True)
class AsyncReleaseDecision:
    decision_id: str
    window_key: AsyncReleaseWindowKey
    rank: int
    action: AsyncDecisionAction
    task_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "window_key": self.window_key.to_dict(),
            "rank": self.rank,
            "action": self.action,
            "task_ids": list(self.task_ids),
            "reason": self.reason,
        }
