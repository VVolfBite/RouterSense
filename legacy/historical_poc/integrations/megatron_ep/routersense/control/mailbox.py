from __future__ import annotations

from dataclasses import dataclass, field

from integrations.megatron_ep.routersense.control.contracts import ControlEnvelope, PendingCommTask
from integrations.megatron_ep.routersense.control.state_machine import transition_task


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
