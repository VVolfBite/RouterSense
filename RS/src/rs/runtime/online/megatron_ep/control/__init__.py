"""控制面子包入口。

控制面负责：
- phase plan agreement
- shadow policy / old control contracts
- 早期 mailbox/state-machine 试验残留
当前正式热路径主要消费 agreement_wire 和 plan_agreement。
"""

from .contracts import (
    BucketDescriptor,
    ControlCommand,
    ControlEnvelope,
    ControlTimelineEvent,
    PendingCommTask,
    PlanExpiry,
    PlanKey,
)
from .mailbox import ControlMailbox, apply_if_pending, expire_if_late
from .plan_agreement import run_phase_plan_agreement
from .state_machine import can_transition, transition_task
from .timeline import ControlTimeline

__all__ = [
    "BucketDescriptor",
    "ControlCommand",
    "ControlEnvelope",
    "ControlMailbox",
    "ControlTimeline",
    "ControlTimelineEvent",
    "PendingCommTask",
    "PlanExpiry",
    "PlanKey",
    "apply_if_pending",
    "can_transition",
    "expire_if_late",
    "run_phase_plan_agreement",
    "transition_task",
]
