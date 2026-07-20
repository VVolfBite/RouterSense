"""早期 control-plane 试验合同。

这里定义 mailbox / envelope / pending task 等控制面结构。
当前更多用于历史测试和兼容，不是主热路径的核心实现。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PlanScope = Literal["phase", "wave", "bucket", "peer_flow"]
ExpiryBoundary = Literal["phase_end", "bucket_id", "wave_id", "explicit_epoch"]
TaskCommitState = Literal[
    "pending",
    "planned",
    "committed",
    "in_flight",
    "completed",
    "expired",
    "fallback_native",
    "failed",
]
PhaseExecutionState = Literal["pending", "planning", "agreed", "executing", "completed", "failed"]


@dataclass(frozen=True)
class PlanKey:
    run_id_digest: str
    forward_epoch: int
    step_id: str
    microbatch_id: str
    layer_id: str
    phase: str
    ep_group_hash: str
    ep_group_epoch: int
    model_revision_hash: str
    expert_placement_hash: str
    request_table_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanExpiry:
    expiry_epoch: int
    expiry_boundary: ExpiryBoundary
    explicit_boundary_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlCommand:
    command_id: str
    plan_key: PlanKey
    plan_hash: str
    policy_name: str
    policy_version: str
    control_mode: str
    phase: str
    action: str
    scope: PlanScope
    issued_epoch: int
    expiry: PlanExpiry
    transport_mutation: bool
    is_shadow_only: bool
    can_preempt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlEnvelope:
    command: ControlCommand
    sender_rank: int
    receiver_group_hash: str
    sequence_no: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BucketDescriptor:
    bucket_id: str
    plan_key: PlanKey
    phase: str
    wave_id: int
    src_rank: int
    dst_rank: int
    source_peer_index: int
    destination_peer_index: int
    source_offset_rows: int
    receive_offset_rows: int
    row_count: int
    byte_count: int
    dtype: str
    hidden_shape_suffix: tuple[int, ...]
    packed_layout_id: str
    segment_ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PendingCommTask:
    task_id: str
    bucket: BucketDescriptor
    release_state: str
    commit_state: TaskCommitState = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlTimelineEvent:
    event: str
    plan_key: PlanKey
    rank: int
    phase_state: PhaseExecutionState
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
