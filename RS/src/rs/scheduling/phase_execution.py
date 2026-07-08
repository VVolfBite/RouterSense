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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PackedTensorDescriptor":
        return cls(
            tensor_role=str(payload["tensor_role"]),
            shape=tuple(int(v) for v in payload["shape"]),
            shape_suffix=tuple(int(v) for v in payload["shape_suffix"]),
            dtype=str(payload["dtype"]),
            device=str(payload["device"]),
            element_size_bytes=int(payload["element_size_bytes"]),
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PayloadSlice":
        return cls(
            bundle_id=str(payload["bundle_id"]),
            tensor_role=str(payload["tensor_role"]),
            src_rank=int(payload["src_rank"]),
            dst_rank=int(payload["dst_rank"]),
            segment_ordinal=int(payload["segment_ordinal"]),
            sender_offset_rows=int(payload["sender_offset_rows"]),
            receiver_offset_rows=int(payload["receiver_offset_rows"]),
            row_count=int(payload["row_count"]),
            dtype=str(payload["dtype"]),
            shape_suffix=tuple(int(v) for v in payload["shape_suffix"]),
            element_size_bytes=int(payload["element_size_bytes"]),
            payload_byte_count=int(payload["payload_byte_count"]),
            packed_layout_id=str(payload["packed_layout_id"]),
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OutgoingSegment":
        return cls(
            segment_id=str(payload["segment_id"]),
            phase=str(payload["phase"]),
            src_rank=int(payload["src_rank"]),
            dst_rank=int(payload["dst_rank"]),
            destination_peer_index=int(payload["destination_peer_index"]),
            segment_ordinal=int(payload["segment_ordinal"]),
            send_offset_rows=int(payload["send_offset_rows"]),
            row_count=int(payload["row_count"]),
            byte_count=int(payload["byte_count"]),
            packed_send_layout_id=str(payload["packed_send_layout_id"]),
            is_local=bool(payload["is_local"]),
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IncomingSlot":
        return cls(
            slot_id=str(payload["slot_id"]),
            phase=str(payload["phase"]),
            src_rank=int(payload["src_rank"]),
            dst_rank=int(payload["dst_rank"]),
            source_peer_index=int(payload["source_peer_index"]),
            segment_ordinal=int(payload["segment_ordinal"]),
            receive_offset_rows=int(payload["receive_offset_rows"]),
            row_count=int(payload["row_count"]),
            byte_count=int(payload["byte_count"]),
            canonical_receive_layout_id=str(payload["canonical_receive_layout_id"]),
            is_local=bool(payload["is_local"]),
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransportBundle":
        return cls(
            bundle_id=str(payload["bundle_id"]),
            phase=str(payload["phase"]),
            atomic_submit=bool(payload["atomic_submit"]),
            outgoing_segment=OutgoingSegment.from_dict(payload["outgoing_segment"]),
            payloads=tuple(PackedTensorDescriptor.from_dict(item) for item in payload["payloads"]),
            payload_slices=tuple(PayloadSlice.from_dict(item) for item in payload["payload_slices"]),
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransferLayout":
        return cls(
            transfer_key=str(payload["transfer_key"]),
            bundle_id=str(payload["bundle_id"]),
            phase=str(payload["phase"]),
            src_rank=int(payload["src_rank"]),
            dst_rank=int(payload["dst_rank"]),
            source_peer_index=int(payload["source_peer_index"]),
            destination_peer_index=int(payload["destination_peer_index"]),
            segment_ordinal=int(payload["segment_ordinal"]),
            sender_offset_rows=int(payload["sender_offset_rows"]),
            receiver_offset_rows=int(payload["receiver_offset_rows"]),
            row_count=int(payload["row_count"]),
            byte_count=int(payload["byte_count"]),
            packed_send_layout_id=str(payload["packed_send_layout_id"]),
            canonical_receive_layout_id=str(payload["canonical_receive_layout_id"]),
            atomic_submit=bool(payload["atomic_submit"]),
            payloads=tuple(PackedTensorDescriptor.from_dict(item) for item in payload["payloads"]),
            payload_slices=tuple(PayloadSlice.from_dict(item) for item in payload["payload_slices"]),
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BucketTask":
        return cls(
            task_id=str(payload["task_id"]),
            bundle_id=str(payload["bundle_id"]),
            phase=str(payload["phase"]),
            src_rank=int(payload["src_rank"]),
            dst_rank=int(payload["dst_rank"]),
            source_peer_index=int(payload["source_peer_index"]),
            destination_peer_index=int(payload["destination_peer_index"]),
            segment_ordinal=int(payload["segment_ordinal"]),
            bucket_ordinal=int(payload["bucket_ordinal"]),
            sender_offset_rows=int(payload["sender_offset_rows"]),
            receiver_offset_rows=int(payload["receiver_offset_rows"]),
            row_count=int(payload["row_count"]),
            byte_count=int(payload["byte_count"]),
            packed_send_layout_id=str(payload["packed_send_layout_id"]),
            canonical_receive_layout_id=str(payload["canonical_receive_layout_id"]),
            payload_slices=tuple(PayloadSlice.from_dict(item) for item in payload["payload_slices"]),
        )


@dataclass(frozen=True)
class PlanWave:
    wave_id: int
    phase: PhaseName
    bucket_tasks: tuple[BucketTask, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanWave":
        return cls(
            wave_id=int(payload["wave_id"]),
            phase=str(payload["phase"]),
            bucket_tasks=tuple(BucketTask.from_dict(item) for item in payload["bucket_tasks"]),
        )


@dataclass(frozen=True)
class FutureDemandHint:
    hint_mode: str = "none"
    hint_digest: str = ""
    hint_source: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FutureDemandHint":
        return cls(
            hint_mode=str(payload.get("hint_mode", "none")),
            hint_digest=str(payload.get("hint_digest", "")),
            hint_source=str(payload.get("hint_source", "none")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_wire_payload(self) -> tuple[str, str, str, dict[str, Any]]:
        return (
            str(self.hint_mode),
            str(self.hint_digest),
            str(self.hint_source),
            dict(self.metadata),
        )

    @classmethod
    def from_wire_payload(cls, payload: tuple[str, str, str, dict[str, Any]] | list[Any]) -> "FutureDemandHint":
        hint_mode, hint_digest, hint_source, metadata = payload
        return cls(
            hint_mode=str(hint_mode),
            hint_digest=str(hint_digest),
            hint_source=str(hint_source),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class PlanningEdgeSummary:
    phase: PhaseName
    src_rank: int
    dst_rank: int
    segment_ordinal: int
    row_count: int
    byte_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanningEdgeSummary":
        return cls(
            phase=str(payload["phase"]),
            src_rank=int(payload["src_rank"]),
            dst_rank=int(payload["dst_rank"]),
            segment_ordinal=int(payload["segment_ordinal"]),
            row_count=int(payload["row_count"]),
            byte_count=int(payload["byte_count"]),
        )

    def to_wire_payload(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.src_rank),
            int(self.dst_rank),
            int(self.segment_ordinal),
            int(self.row_count),
            int(self.byte_count),
        )

    @classmethod
    def from_wire_payload(cls, payload: tuple[int, int, int, int, int] | list[int], *, phase: str) -> "PlanningEdgeSummary":
        src_rank, dst_rank, segment_ordinal, row_count, byte_count = payload
        return cls(
            phase=str(phase),
            src_rank=int(src_rank),
            dst_rank=int(dst_rank),
            segment_ordinal=int(segment_ordinal),
            row_count=int(row_count),
            byte_count=int(byte_count),
        )


@dataclass(frozen=True)
class PhasePlanningSummary:
    plan_key: dict[str, Any]
    phase: PhaseName
    control_mode: str
    layer_id: str
    global_rank: int
    local_rank: int
    ep_group_ranks: tuple[int, ...]
    ep_group_root_rank: int
    per_peer_rows: tuple[int, ...]
    per_peer_bytes: tuple[int, ...]
    outgoing_edges: tuple[PlanningEdgeSummary, ...]
    p2_hint: FutureDemandHint = field(default_factory=FutureDemandHint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_key": dict(self.plan_key),
            "phase": str(self.phase),
            "control_mode": str(self.control_mode),
            "layer_id": str(self.layer_id),
            "global_rank": int(self.global_rank),
            "local_rank": int(self.local_rank),
            "ep_group_ranks": [int(v) for v in self.ep_group_ranks],
            "ep_group_root_rank": int(self.ep_group_root_rank),
            "per_peer_rows": [int(v) for v in self.per_peer_rows],
            "per_peer_bytes": [int(v) for v in self.per_peer_bytes],
            "outgoing_edges": [item.to_dict() for item in self.outgoing_edges],
            "p2_hint": self.p2_hint.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhasePlanningSummary":
        return cls(
            plan_key=dict(payload["plan_key"]),
            phase=str(payload["phase"]),
            control_mode=str(payload["control_mode"]),
            layer_id=str(payload["layer_id"]),
            global_rank=int(payload["global_rank"]),
            local_rank=int(payload["local_rank"]),
            ep_group_ranks=tuple(int(v) for v in payload["ep_group_ranks"]),
            ep_group_root_rank=int(payload["ep_group_root_rank"]),
            per_peer_rows=tuple(int(v) for v in payload["per_peer_rows"]),
            per_peer_bytes=tuple(int(v) for v in payload["per_peer_bytes"]),
            outgoing_edges=tuple(PlanningEdgeSummary.from_dict(item) for item in payload["outgoing_edges"]),
            p2_hint=FutureDemandHint.from_dict(payload.get("p2_hint", {})),
        )

    def to_wire_payload(self) -> tuple[int, tuple[int, ...], tuple[tuple[int, int, int, int, int], ...], tuple[str, str, str, dict[str, Any]]]:
        return (
            int(self.global_rank),
            tuple(int(v) for v in self.per_peer_bytes),
            tuple(item.to_wire_payload() for item in self.outgoing_edges),
            self.p2_hint.to_wire_payload(),
        )

    @classmethod
    def from_wire_payload(
        cls,
        payload: tuple[int, tuple[int, ...], tuple[tuple[int, int, int, int, int], ...], tuple[str, str, str, dict[str, Any]]] | list[Any],
        *,
        phase: str,
        control_mode: str,
        layer_id: str,
        ep_group_ranks: tuple[int, ...],
        ep_group_root_rank: int,
        plan_key_factory,
    ) -> "PhasePlanningSummary":
        global_rank, per_peer_bytes, outgoing_edges, p2_hint = payload
        return cls(
            plan_key=dict(plan_key_factory(int(global_rank))),
            phase=str(phase),
            control_mode=str(control_mode),
            layer_id=str(layer_id),
            global_rank=int(global_rank),
            local_rank=tuple(int(v) for v in ep_group_ranks).index(int(global_rank)),
            ep_group_ranks=tuple(int(v) for v in ep_group_ranks),
            ep_group_root_rank=int(ep_group_root_rank),
            per_peer_rows=tuple(0 for _ in tuple(per_peer_bytes)),
            per_peer_bytes=tuple(int(v) for v in per_peer_bytes),
            outgoing_edges=tuple(PlanningEdgeSummary.from_wire_payload(item, phase=str(phase)) for item in outgoing_edges),
            p2_hint=FutureDemandHint.from_wire_payload(p2_hint),
        )


@dataclass(frozen=True)
class AbstractTaskRef:
    phase: PhaseName
    src_rank: int
    dst_rank: int
    segment_ordinal: int
    bucket_ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AbstractTaskRef":
        return cls(
            phase=str(payload["phase"]),
            src_rank=int(payload["src_rank"]),
            dst_rank=int(payload["dst_rank"]),
            segment_ordinal=int(payload["segment_ordinal"]),
            bucket_ordinal=int(payload["bucket_ordinal"]),
        )

    def to_wire_payload(self) -> tuple[int, int, int, int]:
        return (
            int(self.src_rank),
            int(self.dst_rank),
            int(self.segment_ordinal),
            int(self.bucket_ordinal),
        )

    @classmethod
    def from_wire_payload(cls, payload: tuple[int, int, int, int] | list[int], *, phase: str) -> "AbstractTaskRef":
        src_rank, dst_rank, segment_ordinal, bucket_ordinal = payload
        return cls(
            phase=str(phase),
            src_rank=int(src_rank),
            dst_rank=int(dst_rank),
            segment_ordinal=int(segment_ordinal),
            bucket_ordinal=int(bucket_ordinal),
        )


@dataclass(frozen=True)
class AbstractPlanWave:
    wave_id: int
    phase: PhaseName
    task_refs: tuple[AbstractTaskRef, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_id": int(self.wave_id),
            "phase": str(self.phase),
            "task_refs": [item.to_dict() for item in self.task_refs],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AbstractPlanWave":
        return cls(
            wave_id=int(payload["wave_id"]),
            phase=str(payload["phase"]),
            task_refs=tuple(AbstractTaskRef.from_dict(item) for item in payload["task_refs"]),
        )

    def to_wire_payload(self) -> tuple[int, tuple[tuple[int, int, int, int], ...]]:
        return (
            int(self.wave_id),
            tuple(item.to_wire_payload() for item in self.task_refs),
        )

    @classmethod
    def from_wire_payload(cls, payload: tuple[int, tuple[tuple[int, int, int, int], ...]] | list[Any], *, phase: str) -> "AbstractPlanWave":
        wave_id, task_refs = payload
        return cls(
            wave_id=int(wave_id),
            phase=str(phase),
            task_refs=tuple(AbstractTaskRef.from_wire_payload(item, phase=str(phase)) for item in task_refs),
        )


@dataclass(frozen=True)
class AbstractPhaseExecutionPlan:
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
    waves: tuple[AbstractPlanWave, ...]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_key": dict(self.plan_key),
            "phase": str(self.phase),
            "policy_name": str(self.policy_name),
            "policy_version": str(self.policy_version),
            "control_mode": str(self.control_mode),
            "execution_mode": str(self.execution_mode),
            "transport_mutation": bool(self.transport_mutation),
            "is_shadow_only": bool(self.is_shadow_only),
            "future_hint_mode": str(self.future_hint_mode),
            "root_rank": int(self.root_rank),
            "observation_digest": str(self.observation_digest),
            "plan_hash": str(self.plan_hash),
            "waves": [item.to_dict() for item in self.waves],
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AbstractPhaseExecutionPlan":
        return cls(
            plan_key=dict(payload["plan_key"]),
            phase=str(payload["phase"]),
            policy_name=str(payload["policy_name"]),
            policy_version=str(payload["policy_version"]),
            control_mode=str(payload["control_mode"]),
            execution_mode=str(payload["execution_mode"]),
            transport_mutation=bool(payload["transport_mutation"]),
            is_shadow_only=bool(payload["is_shadow_only"]),
            future_hint_mode=str(payload["future_hint_mode"]),
            root_rank=int(payload["root_rank"]),
            observation_digest=str(payload["observation_digest"]),
            plan_hash=str(payload["plan_hash"]),
            waves=tuple(AbstractPlanWave.from_dict(item) for item in payload["waves"]),
            metrics=dict(payload.get("metrics", {}) or {}),
        )

    def to_wire_payload(self, *, minimal_metrics: bool = False) -> tuple[
        dict[str, Any],
        str,
        str,
        str,
        str,
        str,
        bool,
        bool,
        str,
        int,
        str,
        str,
        tuple[tuple[int, tuple[tuple[int, int, int, int], ...]], ...],
        dict[str, Any],
    ]:
        return (
            dict(self.plan_key),
            str(self.phase),
            str(self.policy_name),
            str(self.policy_version),
            str(self.control_mode),
            str(self.execution_mode),
            bool(self.transport_mutation),
            bool(self.is_shadow_only),
            str(self.future_hint_mode),
            int(self.root_rank),
            str(self.observation_digest),
            str(self.plan_hash),
            tuple(wave.to_wire_payload() for wave in self.waves),
            (
                {
                    "bucket_rows": int(self.metrics.get("bucket_rows", 0) or 0),
                    "wave_count": int(self.metrics.get("wave_count", len(self.waves)) or len(self.waves)),
                    "transport_mutation": bool(self.metrics.get("transport_mutation", self.transport_mutation)),
                }
                if minimal_metrics
                else dict(self.metrics)
            ),
        )

    @classmethod
    def from_wire_payload(cls, payload) -> "AbstractPhaseExecutionPlan":
        (
            plan_key,
            phase,
            policy_name,
            policy_version,
            control_mode,
            execution_mode,
            transport_mutation,
            is_shadow_only,
            future_hint_mode,
            root_rank,
            observation_digest,
            plan_hash,
            waves,
            metrics,
        ) = payload
        return cls(
            plan_key=dict(plan_key),
            phase=str(phase),
            policy_name=str(policy_name),
            policy_version=str(policy_version),
            control_mode=str(control_mode),
            execution_mode=str(execution_mode),
            transport_mutation=bool(transport_mutation),
            is_shadow_only=bool(is_shadow_only),
            future_hint_mode=str(future_hint_mode),
            root_rank=int(root_rank),
            observation_digest=str(observation_digest),
            plan_hash=str(plan_hash),
            waves=tuple(AbstractPlanWave.from_wire_payload(item, phase=str(phase)) for item in waves),
            metrics=dict(metrics or {}),
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhaseExecutionPlan":
        return cls(
            plan_key=dict(payload["plan_key"]),
            phase=str(payload["phase"]),
            policy_name=str(payload["policy_name"]),
            policy_version=str(payload["policy_version"]),
            control_mode=str(payload["control_mode"]),
            execution_mode=str(payload["execution_mode"]),
            transport_mutation=bool(payload["transport_mutation"]),
            is_shadow_only=bool(payload["is_shadow_only"]),
            future_hint_mode=str(payload["future_hint_mode"]),
            root_rank=int(payload["root_rank"]),
            observation_digest=str(payload["observation_digest"]),
            plan_hash=str(payload["plan_hash"]),
            waves=tuple(PlanWave.from_dict(item) for item in payload["waves"]),
            metrics=dict(payload.get("metrics", {}) or {}),
        )

    def to_abstract_plan(self) -> AbstractPhaseExecutionPlan:
        return AbstractPhaseExecutionPlan(
            plan_key=dict(self.plan_key),
            phase=str(self.phase),
            policy_name=str(self.policy_name),
            policy_version=str(self.policy_version),
            control_mode=str(self.control_mode),
            execution_mode=str(self.execution_mode),
            transport_mutation=bool(self.transport_mutation),
            is_shadow_only=bool(self.is_shadow_only),
            future_hint_mode=str(self.future_hint_mode),
            root_rank=int(self.root_rank),
            observation_digest=str(self.observation_digest),
            plan_hash=str(self.plan_hash),
            waves=tuple(
                AbstractPlanWave(
                    wave_id=int(wave.wave_id),
                    phase=str(wave.phase),
                    task_refs=tuple(
                        AbstractTaskRef(
                            phase=str(task.phase),
                            src_rank=int(task.src_rank),
                            dst_rank=int(task.dst_rank),
                            segment_ordinal=int(task.segment_ordinal),
                            bucket_ordinal=int(task.bucket_ordinal),
                        )
                        for task in wave.bucket_tasks
                    ),
                )
                for wave in self.waves
            ),
            metrics=dict(self.metrics),
        )


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
    payload_specs: tuple[PackedTensorDescriptor, ...]
    atomic_submit: bool
    outgoing_segments: tuple[OutgoingSegment, ...]
    incoming_slots: tuple[IncomingSlot, ...]
    transport_bundles: tuple[TransportBundle, ...]
    release_state: str
    demand_known_at: str
    payload_exists: bool
    p2_hint: FutureDemandHint = field(default_factory=FutureDemandHint)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_planning_summary(self) -> PhasePlanningSummary:
        return PhasePlanningSummary(
            plan_key=dict(self.plan_key),
            phase=str(self.phase),
            control_mode=str(self.control_mode),
            layer_id=str(self.layer_id),
            global_rank=int(self.global_rank),
            local_rank=int(self.local_rank),
            ep_group_ranks=tuple(int(v) for v in self.ep_group_ranks),
            ep_group_root_rank=int(self.ep_group_root_rank),
            per_peer_rows=tuple(int(v) for v in self.per_peer_rows),
            per_peer_bytes=tuple(int(v) for v in self.per_peer_bytes),
            outgoing_edges=tuple(
                PlanningEdgeSummary(
                    phase=str(segment.phase),
                    src_rank=int(segment.src_rank),
                    dst_rank=int(segment.dst_rank),
                    segment_ordinal=int(segment.segment_ordinal),
                    row_count=int(segment.row_count),
                    byte_count=int(segment.byte_count),
                )
                for segment in self.outgoing_segments
                if not bool(segment.is_local) and int(segment.row_count) > 0
            ),
            p2_hint=self.p2_hint,
        )

    def _agreement_payload_specs(self) -> list[dict[str, Any]]:
        payloads = self.payload_specs
        return [
            {
                "tensor_role": str(payload.tensor_role),
                "dtype": str(payload.dtype),
                "shape_suffix": [int(v) for v in payload.shape_suffix],
                "element_size_bytes": int(payload.element_size_bytes),
            }
            for payload in payloads
        ]

    def to_agreement_payload(self) -> dict[str, Any]:
        return {
            "plan_key": dict(self.plan_key),
            "phase": str(self.phase),
            "control_mode": str(self.control_mode),
            "forward_epoch": int(self.forward_epoch),
            "layer_id": str(self.layer_id),
            "layer_name": str(self.layer_name),
            "global_rank": int(self.global_rank),
            "local_rank": int(self.local_rank),
            "ep_group_ranks": [int(v) for v in self.ep_group_ranks],
            "ep_group_root_rank": int(self.ep_group_root_rank),
            "topology": dict(self.topology),
            "dispatcher_class": str(self.dispatcher_class),
            "dispatcher_fingerprint": dict(self.dispatcher_fingerprint),
            "expert_placement_hash": str(self.expert_placement_hash),
            "input_splits": [int(v) for v in self.input_splits],
            "output_splits": [int(v) for v in self.output_splits],
            "send_splits": [int(v) for v in self.send_splits],
            "recv_splits": [int(v) for v in self.recv_splits],
            "per_peer_rows": [int(v) for v in self.per_peer_rows],
            "per_peer_bytes": [int(v) for v in self.per_peer_bytes],
            "packed_send_layout_id": str(self.packed_send_layout_id),
            "canonical_receive_layout_id": str(self.canonical_receive_layout_id),
            "outgoing_segments": [
                {
                    "segment_id": str(segment.segment_id),
                    "phase": str(segment.phase),
                    "src_rank": int(segment.src_rank),
                    "dst_rank": int(segment.dst_rank),
                    "destination_peer_index": int(segment.destination_peer_index),
                    "segment_ordinal": int(segment.segment_ordinal),
                    "send_offset_rows": int(segment.send_offset_rows),
                    "row_count": int(segment.row_count),
                    "byte_count": int(segment.byte_count),
                    "packed_send_layout_id": str(segment.packed_send_layout_id),
                    "is_local": bool(segment.is_local),
                }
                for segment in self.outgoing_segments
            ],
            "incoming_slots": [
                {
                    "slot_id": str(slot.slot_id),
                    "phase": str(slot.phase),
                    "src_rank": int(slot.src_rank),
                    "dst_rank": int(slot.dst_rank),
                    "source_peer_index": int(slot.source_peer_index),
                    "segment_ordinal": int(slot.segment_ordinal),
                    "receive_offset_rows": int(slot.receive_offset_rows),
                    "row_count": int(slot.row_count),
                    "byte_count": int(slot.byte_count),
                    "canonical_receive_layout_id": str(slot.canonical_receive_layout_id),
                    "is_local": bool(slot.is_local),
                }
                for slot in self.incoming_slots
            ],
            "payload_specs": self._agreement_payload_specs(),
            "atomic_submit": bool(self.atomic_submit),
            "release_state": str(self.release_state),
            "demand_known_at": str(self.demand_known_at),
            "payload_exists": bool(self.payload_exists),
            "p2_hint": self.p2_hint.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhaseReadyContext":
        transport_bundles = tuple(TransportBundle.from_dict(item) for item in payload["transport_bundles"])
        payload_specs = tuple(
            PackedTensorDescriptor.from_dict(item)
            for item in payload.get("payload_specs", [])
        )
        if not payload_specs and transport_bundles:
            payload_specs = tuple(transport_bundles[0].payloads)
        return cls(
            plan_key=dict(payload["plan_key"]),
            phase=str(payload["phase"]),
            control_mode=str(payload["control_mode"]),
            forward_epoch=int(payload["forward_epoch"]),
            layer_id=str(payload["layer_id"]),
            layer_name=str(payload["layer_name"]),
            global_rank=int(payload["global_rank"]),
            local_rank=int(payload["local_rank"]),
            ep_group_ranks=tuple(int(v) for v in payload["ep_group_ranks"]),
            ep_group_root_rank=int(payload["ep_group_root_rank"]),
            topology=dict(payload["topology"]),
            dispatcher_class=str(payload["dispatcher_class"]),
            dispatcher_fingerprint=dict(payload["dispatcher_fingerprint"]),
            expert_placement_hash=str(payload["expert_placement_hash"]),
            input_splits=tuple(int(v) for v in payload["input_splits"]),
            output_splits=tuple(int(v) for v in payload["output_splits"]),
            send_splits=tuple(int(v) for v in payload["send_splits"]),
            recv_splits=tuple(int(v) for v in payload["recv_splits"]),
            per_peer_rows=tuple(int(v) for v in payload["per_peer_rows"]),
            per_peer_bytes=tuple(int(v) for v in payload["per_peer_bytes"]),
            packed_send_layout_id=str(payload["packed_send_layout_id"]),
            canonical_receive_layout_id=str(payload["canonical_receive_layout_id"]),
            payload_specs=payload_specs,
            atomic_submit=bool(payload.get("atomic_submit", transport_bundles[0].atomic_submit if transport_bundles else True)),
            outgoing_segments=tuple(OutgoingSegment.from_dict(item) for item in payload["outgoing_segments"]),
            incoming_slots=tuple(IncomingSlot.from_dict(item) for item in payload["incoming_slots"]),
            transport_bundles=transport_bundles,
            release_state=str(payload["release_state"]),
            demand_known_at=str(payload["demand_known_at"]),
            payload_exists=bool(payload["payload_exists"]),
            p2_hint=FutureDemandHint.from_dict(payload.get("p2_hint", {})),
        )

    @classmethod
    def from_agreement_payload(cls, payload: dict[str, Any]) -> "PhaseReadyContext":
        def _payload_descriptors_from_specs(specs: list[dict[str, Any]]) -> tuple[PackedTensorDescriptor, ...]:
            descriptors: list[PackedTensorDescriptor] = []
            for item in specs:
                role = str(item["tensor_role"])
                descriptors.append(
                    PackedTensorDescriptor(
                        tensor_role=role,
                        shape=(),
                        shape_suffix=tuple(int(v) for v in item.get("shape_suffix", [])),
                        dtype=str(item["dtype"]),
                        device="",
                        element_size_bytes=int(item["element_size_bytes"]),
                    )
                )
            return tuple(descriptors)

        payload_descriptors = _payload_descriptors_from_specs(list(payload.get("payload_specs", []) or []))
        return cls(
            plan_key=dict(payload["plan_key"]),
            phase=str(payload["phase"]),
            control_mode=str(payload["control_mode"]),
            forward_epoch=int(payload["forward_epoch"]),
            layer_id=str(payload["layer_id"]),
            layer_name=str(payload["layer_name"]),
            global_rank=int(payload["global_rank"]),
            local_rank=int(payload["local_rank"]),
            ep_group_ranks=tuple(int(v) for v in payload["ep_group_ranks"]),
            ep_group_root_rank=int(payload["ep_group_root_rank"]),
            topology=dict(payload["topology"]),
            dispatcher_class=str(payload["dispatcher_class"]),
            dispatcher_fingerprint=dict(payload["dispatcher_fingerprint"]),
            expert_placement_hash=str(payload["expert_placement_hash"]),
            input_splits=tuple(int(v) for v in payload["input_splits"]),
            output_splits=tuple(int(v) for v in payload["output_splits"]),
            send_splits=tuple(int(v) for v in payload["send_splits"]),
            recv_splits=tuple(int(v) for v in payload["recv_splits"]),
            per_peer_rows=tuple(int(v) for v in payload["per_peer_rows"]),
            per_peer_bytes=tuple(int(v) for v in payload["per_peer_bytes"]),
            packed_send_layout_id=str(payload["packed_send_layout_id"]),
            canonical_receive_layout_id=str(payload["canonical_receive_layout_id"]),
            payload_specs=payload_descriptors,
            atomic_submit=bool(payload.get("atomic_submit", True)),
            outgoing_segments=tuple(OutgoingSegment.from_dict(item) for item in payload["outgoing_segments"]),
            incoming_slots=tuple(IncomingSlot.from_dict(item) for item in payload["incoming_slots"]),
            transport_bundles=(),
            release_state=str(payload["release_state"]),
            demand_known_at=str(payload["demand_known_at"]),
            payload_exists=bool(payload["payload_exists"]),
            p2_hint=FutureDemandHint.from_dict(payload.get("p2_hint", {})),
        )

    @classmethod
    def from_planning_summary(cls, summary: PhasePlanningSummary) -> "PhaseReadyContext":
        return cls(
            plan_key=dict(summary.plan_key),
            phase=str(summary.phase),
            control_mode=str(summary.control_mode),
            forward_epoch=0,
            layer_id=str(summary.layer_id),
            layer_name=str(summary.layer_id),
            global_rank=int(summary.global_rank),
            local_rank=int(summary.local_rank),
            ep_group_ranks=tuple(int(v) for v in summary.ep_group_ranks),
            ep_group_root_rank=int(summary.ep_group_root_rank),
            topology={},
            dispatcher_class="",
            dispatcher_fingerprint={},
            expert_placement_hash="",
            input_splits=tuple(),
            output_splits=tuple(),
            send_splits=tuple(),
            recv_splits=tuple(),
            per_peer_rows=tuple(int(v) for v in summary.per_peer_rows),
            per_peer_bytes=tuple(int(v) for v in summary.per_peer_bytes),
            packed_send_layout_id="",
            canonical_receive_layout_id="",
            payload_specs=tuple(),
            atomic_submit=True,
            outgoing_segments=tuple(
                OutgoingSegment(
                    segment_id=f"{item.phase}:{item.src_rank}->{item.dst_rank}:{item.segment_ordinal}",
                    phase=str(item.phase),
                    src_rank=int(item.src_rank),
                    dst_rank=int(item.dst_rank),
                    destination_peer_index=-1,
                    segment_ordinal=int(item.segment_ordinal),
                    send_offset_rows=0,
                    row_count=int(item.row_count),
                    byte_count=int(item.byte_count),
                    packed_send_layout_id="",
                    is_local=False,
                )
                for item in summary.outgoing_edges
            ),
            incoming_slots=tuple(),
            transport_bundles=(),
            release_state="ready",
            demand_known_at="runtime",
            payload_exists=True,
            p2_hint=summary.p2_hint,
        )


@dataclass(frozen=True)
class PhaseHookResult:
    accepted: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
