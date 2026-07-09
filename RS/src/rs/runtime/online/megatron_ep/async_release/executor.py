"""Fail-closed async-release executor framework.

The real collective path is disabled by default. This module only prepares a
compiled schedule and, unless explicitly enabled, returns a phase_sync
fallback plan with diagnostics suitable for debug/export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .agreement import validate_async_release_global_agreement
from .compiled_schedule import CompiledAsyncReleaseSchedule, compile_async_release_schedule
from .contracts import AsyncReleaseExecutionPlan, AsyncReleasePreparedPlan
from .plan_builder import validate_async_release_execution_plan


@dataclass(frozen=True)
class AsyncReleaseExecutorConfig:
    enabled: bool = False
    dry_run: bool = True
    allow_real_collectives: bool = False
    require_global_order_agreement: bool = True
    fallback_policy: str = "phase_sync"
    max_ready_queue_size: int = 4096
    debug_artifacts: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AsyncReleaseExecutor:
    def __init__(self, *, config: AsyncReleaseExecutorConfig) -> None:
        self.config = config

    def prepare(
        self,
        plan: AsyncReleaseExecutionPlan,
        *,
        rank: int,
        world_size: int,
    ) -> tuple[AsyncReleasePreparedPlan, CompiledAsyncReleaseSchedule]:
        compiled = compile_async_release_schedule(plan, device="cpu")
        validation = validate_async_release_execution_plan(plan)
        ready_task_count = sum(1 for task in plan.phase_tasks if str(task.get("release_dependency", "none")) == "none")
        blocked_task_count = max(0, int(len(plan.phase_tasks)) - int(ready_task_count))
        fallback_reasons: list[str] = []
        agreement_ok = True
        if not self.config.enabled:
            fallback_reasons.append("executor_disabled")
        if self.config.max_ready_queue_size > 0 and int(compiled.task_count) > int(self.config.max_ready_queue_size):
            fallback_reasons.append("task_count_exceeds_max_ready_queue_size")
        if not validation["valid"]:
            fallback_reasons.append("dependency_validation_failed")
        if self.config.require_global_order_agreement:
            agreement = validate_async_release_global_agreement((compiled,))
            agreement_ok = bool(agreement["valid"])
            if not agreement_ok:
                fallback_reasons.append("global_order_agreement_failed")
        prepared = AsyncReleasePreparedPlan(
            plan_id=str(plan.plan_id),
            compiled_schedule_digest=str(compiled.digest),
            compiled_schedule_task_count=int(compiled.task_count),
            global_order_agreement_required=bool(self.config.require_global_order_agreement),
            global_order_agreement_passed=bool(agreement_ok),
            dependency_validation_passed=bool(validation["valid"]),
            fallback_to_phase_sync=bool(fallback_reasons),
            fallback_reason=",".join(fallback_reasons) if fallback_reasons else "",
            ready_task_count=int(ready_task_count),
            blocked_task_count=int(blocked_task_count),
            dependency_violation_count=int(len(validation["errors"])),
            debug_replay_only=bool(self.config.dry_run or not self.config.allow_real_collectives),
        )
        return prepared, compiled

    def execute_or_fallback(
        self,
        plan: AsyncReleaseExecutionPlan,
        *,
        rank: int,
        world_size: int,
    ) -> dict[str, Any]:
        prepared, compiled = self.prepare(plan, rank=rank, world_size=world_size)
        fallback_reason = prepared.fallback_reason
        real_collective_allowed = (
            bool(self.config.enabled)
            and not bool(self.config.dry_run)
            and bool(self.config.allow_real_collectives)
            and bool(prepared.global_order_agreement_passed)
            and bool(prepared.dependency_validation_passed)
            and not bool(prepared.fallback_to_phase_sync)
        )
        if not bool(self.config.allow_real_collectives):
            fallback_reason = fallback_reason or "allow_real_collectives_disabled"
        elif bool(self.config.dry_run):
            fallback_reason = fallback_reason or "dry_run_enabled"
        elif not bool(prepared.global_order_agreement_passed):
            fallback_reason = fallback_reason or "global_order_agreement_failed"
        elif not bool(prepared.dependency_validation_passed):
            fallback_reason = fallback_reason or "dependency_validation_failed"
        result = {
            "async_release_enabled": bool(self.config.enabled),
            "dry_run": bool(self.config.dry_run),
            "allow_real_collectives": bool(self.config.allow_real_collectives),
            "async_release_real_collectives": bool(real_collective_allowed),
            "fallback_to_phase_sync": not bool(real_collective_allowed),
            "fallback_policy": str(self.config.fallback_policy),
            "fallback_reason": str(fallback_reason),
            "global_order_agreement_passed": bool(prepared.global_order_agreement_passed),
            "dependency_validation_passed": bool(prepared.dependency_validation_passed),
            "compiled_schedule_digest": str(compiled.digest),
            "compiled_schedule_task_count": int(compiled.task_count),
            "task_count": int(compiled.task_count),
            "ready_task_count": int(prepared.ready_task_count),
            "blocked_task_count": int(prepared.blocked_task_count),
            "dependency_violation_count": int(prepared.dependency_violation_count),
            "debug_replay_only": bool(prepared.debug_replay_only),
            "rank": int(rank),
            "world_size": int(world_size),
            "execution_events": (),
        }
        return result


__all__ = ["AsyncReleaseExecutor", "AsyncReleaseExecutorConfig"]
