"""早期 mailbox 控制模型。

主要提供：
- ControlMailbox
- apply_if_pending / expire_if_late
当前主要服务于历史控制面测试，不是正式主线执行路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rs.runtime.online.megatron_ep.control.contracts import ControlEnvelope, PendingCommTask
from rs.runtime.online.megatron_ep.control.state_machine import transition_task


@dataclass
class ControlMailbox:
    envelopes: list[ControlEnvelope] = field(default_factory=list)

    def submit(self, envelope: ControlEnvelope) -> None:
        self.envelopes.append(envelope)
        self.envelopes.sort(key=lambda item: item.sequence_no)

    def poll_control_command(self) -> ControlEnvelope | None:
        if not self.envelopes:
            return None
        return self.envelopes.pop(0)


def expire_if_late(task: PendingCommTask, *, current_epoch: int, expiry_epoch: int) -> PendingCommTask:
    if current_epoch > expiry_epoch and task.commit_state in {"pending", "planned"}:
        return transition_task(task, "expired")
    return task


def apply_if_pending(task: PendingCommTask) -> bool:
    return task.commit_state in {"pending", "planned"}
