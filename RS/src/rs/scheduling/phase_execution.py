"""Shared phase-execution contracts for scheduling and online runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PhaseName = Literal["P0", "P1"]


@dataclass(frozen=True)
class PackedTensorDescriptor:
    tensor_role: str
    shape: tuple[int, ...]
    shape_suffix: tuple[int, ...]
    dtype: str
    device: str
    element_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PayloadSlice:
    bundle_id: str
    tensor_role: str
    src_rank: int
    dst_rank: int
    segment_ordinal: int
    sender_offset_rows: int
    receiver_offset_rows: int
    row_count: int
    dtype: str
    shape_suffix: tuple[int, ...]
    element_size_bytes: int
    payload_byte_count: int
    packed_layout_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutgoingSegment:
    segment_id: str
    phase: PhaseName
    src_rank: int
    dst_rank: int
    destination_peer_index: int
    segment_ordinal: int
    send_offset_rows: int
    row_count: int
    byte_count: int
    packed_send_layout_id: str
    is_local: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IncomingSlot:
    slot_id: str
    phase: PhaseName
    src_rank: int
    dst_rank: int
    source_peer_index: int
    segment_ordinal: int
    receive_offset_rows: int
    row_count: int
    byte_count: int
    canonical_receive_layout_id: str
    is_local: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransportBundle:
    bundle_id: str
    phase: PhaseName
    atomic_submit: bool
    outgoing_segment: OutgoingSegment
    payloads: tuple[PackedTensorDescriptor, ...]
    payload_slices: tuple[PayloadSlice, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransferLayout:
    transfer_key: str
    bundle_id: str
    phase: PhaseName
    src_rank: int
    dst_rank: int
    source_peer_index: int
    destination_peer_index: int
    segment_ordinal: int
    sender_offset_rows: int
    receiver_offset_rows: int
    row_count: int
    byte_count: int
    packed_send_layout_id: str
    canonical_receive_layout_id: str
    atomic_submit: bool
    payloads: tuple[PackedTensorDescriptor, ...]
    payload_slices: tuple[PayloadSlice, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BucketTask:
    task_id: str
    bundle_id: str
    phase: PhaseName
    src_rank: int
    dst_rank: int
    source_peer_index: int
    destination_peer_index: int
    segment_ordinal: int
    bucket_ordinal: int
    sender_offset_rows: int
    receiver_offset_rows: int
    row_count: int
    byte_count: int
    packed_send_layout_id: str
    canonical_receive_layout_id: str
    payload_slices: tuple[PayloadSlice, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanWave:
    wave_id: int
    phase: PhaseName
    bucket_tasks: tuple[BucketTask, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FutureDemandHint:
    hint_mode: str = "none"
    hint_digest: str = ""
    hint_source: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseExecutionPlan:
    plan_key: dict[str, Any]
    phase: PhaseName
    policy_name: str
    policy_version: str
    control_mode: str
    execution_mode: str
    transport_mutation: bool
    is_shadow_only: bool
    future_hint_mode: str
    root_rank: int
    observation_digest: str
    plan_hash: str
    waves: tuple[PlanWave, ...]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseReadyContext:
    plan_key: dict[str, Any]
    phase: PhaseName
    control_mode: str
    forward_epoch: int
    layer_id: str
    layer_name: str
    global_rank: int
    local_rank: int
    ep_group_ranks: tuple[int, ...]
    ep_group_root_rank: int
    topology: dict[str, Any]
    dispatcher_class: str
    dispatcher_fingerprint: dict[str, Any]
    expert_placement_hash: str
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    per_peer_rows: tuple[int, ...]
    per_peer_bytes: tuple[int, ...]
    packed_send_layout_id: str
    canonical_receive_layout_id: str
    outgoing_segments: tuple[OutgoingSegment, ...]
    incoming_slots: tuple[IncomingSlot, ...]
    transport_bundles: tuple[TransportBundle, ...]
    release_state: str
    demand_known_at: str
    payload_exists: bool
    p2_hint: FutureDemandHint = field(default_factory=FutureDemandHint)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseHookResult:
    accepted: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
