"""早期 pending task 状态机。

主要函数：
- can_transition()
- transition_task()
目前更多用于 contract tests 和历史控制模型验证。
"""

from __future__ import annotations

from rs.runtime.online.megatron_ep.control.contracts import PendingCommTask, TaskCommitState


_ALLOWED_TRANSITIONS: dict[TaskCommitState, set[TaskCommitState]] = {
    "pending": {"planned", "expired", "failed", "fallback_native"},
    "planned": {"committed", "expired", "failed", "fallback_native"},
    "committed": {"in_flight", "failed"},
    "in_flight": {"completed", "failed"},
    "completed": set(),
    "expired": set(),
    "fallback_native": {"completed", "failed"},
    "failed": set(),
}


def can_transition(from_state: TaskCommitState, to_state: TaskCommitState) -> bool:
    return to_state in _ALLOWED_TRANSITIONS[from_state]


def transition_task(task: PendingCommTask, to_state: TaskCommitState) -> PendingCommTask:
    if not can_transition(task.commit_state, to_state):
        raise ValueError(f"invalid task transition {task.commit_state}->{to_state} for task_id={task.task_id}")
    return PendingCommTask(
        task_id=task.task_id,
        bucket=task.bucket,
        release_state=task.release_state,
        commit_state=to_state,
    )
