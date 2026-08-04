from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TypeAlias

TimeNs: TypeAlias = int
TaskId: TypeAlias = str
PlanId: TypeAlias = str
StableDigest: TypeAlias = str


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_nonnegative(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")


def _validated_ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    for value in values:
        _require_nonempty(name, value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


class PhaseKind(str, Enum):
    DISPATCH = "DISPATCH"
    COMBINE = "COMBINE"


class ExpectationOrigin(str, Enum):
    DISPATCH_DESCRIPTOR = "DISPATCH_DESCRIPTOR"
    COMBINE_REALIZED = "COMBINE_REALIZED"


class PlanStatus(str, Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"


class TaskState(str, Enum):
    PENDING_DEPENDENCY = "PENDING_DEPENDENCY"
    READY_UNCOMMITTED = "READY_UNCOMMITTED"
    COMMITTED = "COMMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"



class LinkClass(str, Enum):
    LOCAL_ASSEMBLY = "LOCAL_ASSEMBLY"
    INTRA_NODE = "INTRA_NODE"
    INTER_NODE = "INTER_NODE"


class SubmitOutcome(str, Enum):
    PREPARED = "PREPARED"
    RETRYABLE_RESOURCE_BUSY = "RETRYABLE_RESOURCE_BUSY"
    RETRYABLE_STALE_AUTHORITY = "RETRYABLE_STALE_AUTHORITY"
    FATAL_CONTRACT_ERROR = "FATAL_CONTRACT_ERROR"


@dataclass(frozen=True, slots=True)
class AuthorityStamp:
    """Opaque Scheduler authority echoed by transport without token parsing."""

    phase_token: str
    plan_id: PlanId
    phase_plan_epoch: int
    authority_digest: StableDigest

    def __post_init__(self) -> None:
        for name in ("phase_token", "plan_id", "authority_digest"):
            _require_nonempty(name, getattr(self, name))
        _require_nonnegative("phase_plan_epoch", self.phase_plan_epoch)


@dataclass(frozen=True, slots=True)
class NetworkTopology:
    """Immutable topology truth shared by transport, scheduler, and runtime."""

    topology_id: str
    topology_digest: StableDigest
    rank_to_node: tuple[int, ...]
    tx_nic_id_by_rank: tuple[str, ...]
    rx_nic_id_by_rank: tuple[str, ...]
    lane_ids_by_link_class: tuple[tuple[LinkClass, tuple[str, ...]], ...]
    nic_id_by_lane: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_nonempty("topology_id", self.topology_id)
        _require_nonempty("topology_digest", self.topology_digest)
        if not isinstance(self.rank_to_node, tuple) or not self.rank_to_node:
            raise TypeError("rank_to_node must be a non-empty tuple")
        for node_id in self.rank_to_node:
            _require_nonnegative("rank_to_node", node_id)
        world_size = len(self.rank_to_node)
        for name in ("tx_nic_id_by_rank", "rx_nic_id_by_rank"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != world_size:
                raise ValueError(f"{name} must have world_size entries")
            for value in values:
                _require_nonempty(name, value)
        if not isinstance(self.lane_ids_by_link_class, tuple):
            raise TypeError("lane_ids_by_link_class must be tuple")
        seen_classes: set[LinkClass] = set()
        all_lanes: set[str] = set()
        for link_class, lane_ids in self.lane_ids_by_link_class:
            if not isinstance(link_class, LinkClass):
                raise TypeError("lane link class must be LinkClass")
            if link_class is LinkClass.LOCAL_ASSEMBLY:
                raise ValueError("LOCAL_ASSEMBLY has no DataPlane lanes")
            if link_class in seen_classes:
                raise ValueError("duplicate link class in topology")
            seen_classes.add(link_class)
            _validated_ids(lane_ids, "lane_ids")
            overlap = all_lanes & set(lane_ids)
            if overlap:
                raise ValueError(f"lane IDs reused across link classes: {sorted(overlap)}")
            all_lanes.update(lane_ids)
        if not isinstance(self.nic_id_by_lane, tuple):
            raise TypeError("nic_id_by_lane must be tuple")
        lane_map: dict[str, str] = {}
        for lane_id, nic_id in self.nic_id_by_lane:
            _require_nonempty("lane_id", lane_id)
            _require_nonempty("nic_id", nic_id)
            if lane_id in lane_map:
                raise ValueError("duplicate lane in nic_id_by_lane")
            lane_map[lane_id] = nic_id
        if set(lane_map) != all_lanes:
            raise ValueError("nic_id_by_lane must cover every and only declared lane")

    @property
    def world_size(self) -> int:
        return len(self.rank_to_node)


@dataclass(frozen=True, slots=True)
class TaskResourceFootprint:
    task_id: TaskId
    topology_digest: StableDigest
    link_class: LinkClass
    src_resource_id: str
    dst_resource_id: str
    tx_nic_id: str
    rx_nic_id: str
    eligible_lane_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "task_id", "topology_digest", "src_resource_id", "dst_resource_id",
            "tx_nic_id", "rx_nic_id",
        ):
            _require_nonempty(name, getattr(self, name))
        if not isinstance(self.link_class, LinkClass):
            raise TypeError("link_class must be LinkClass")
        if self.link_class is LinkClass.LOCAL_ASSEMBLY:
            raise ValueError("local assembly has no DataPlane footprint")
        object.__setattr__(
            self, "eligible_lane_ids", _validated_ids(self.eligible_lane_ids, "eligible_lane_ids")
        )
        if not self.eligible_lane_ids:
            raise ValueError("eligible_lane_ids must be non-empty")


@dataclass(frozen=True, slots=True)
class ControlPlaneProfile:
    profile_id: str
    profile_digest: StableDigest
    profile_provenance: str
    performance_eligible: bool
    fixed_latency_ns: int
    bandwidth_bytes_per_second: int
    channel_count: int = 1
    fifo: bool = True
    non_preemptive: bool = True
    shares_data_nic: bool = False

    def __post_init__(self) -> None:
        for name in ("profile_id", "profile_digest", "profile_provenance"):
            _require_nonempty(name, getattr(self, name))
        if not isinstance(self.performance_eligible, bool):
            raise TypeError("performance_eligible must be bool")
        _require_nonnegative("fixed_latency_ns", self.fixed_latency_ns)
        if not isinstance(self.bandwidth_bytes_per_second, int) or isinstance(self.bandwidth_bytes_per_second, bool) or self.bandwidth_bytes_per_second <= 0:
            raise ValueError("bandwidth_bytes_per_second must be positive")
        if not isinstance(self.channel_count, int) or isinstance(self.channel_count, bool) or self.channel_count <= 0:
            raise ValueError("channel_count must be positive")
        if self.channel_count != 1 or not self.fifo or not self.non_preemptive or self.shares_data_nic:
            raise ValueError("ControlPlane is one explicit channel independent of DataPlane NIC resources")


@dataclass(frozen=True, slots=True)
class ExactRowDescriptor:
    phase_key: "PhaseKey"
    src_rank: int
    realized_rows_by_destination: tuple[int, ...]
    payload_bytes_by_destination: tuple[int, ...]
    payload_spec_digest: StableDigest
    descriptor_digest: StableDigest
    published_at_ns: TimeNs
    descriptor_payload_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        if self.phase_key.phase_kind is not PhaseKind.DISPATCH:
            raise ValueError("ExactRowDescriptor is Dispatch-only")
        _require_nonnegative("src_rank", self.src_rank)
        for name in ("realized_rows_by_destination", "payload_bytes_by_destination"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise TypeError(f"{name} must be a non-empty tuple")
            for value in values:
                _require_nonnegative(name, value)
        if len(self.realized_rows_by_destination) != len(self.payload_bytes_by_destination):
            raise ValueError("descriptor rows and payload bytes must have equal destination count")
        for rows, payload_bytes in zip(self.realized_rows_by_destination, self.payload_bytes_by_destination):
            if rows == 0 and payload_bytes != 0:
                raise ValueError("zero-row descriptor edge must have zero payload bytes")
            if rows > 0 and payload_bytes <= 0:
                raise ValueError("nonzero-row descriptor edge must have positive payload bytes")
        _require_nonempty("payload_spec_digest", self.payload_spec_digest)
        _require_nonempty("descriptor_digest", self.descriptor_digest)
        _require_nonnegative("published_at_ns", self.published_at_ns)
        _require_nonnegative("descriptor_payload_bytes", self.descriptor_payload_bytes)


@dataclass(frozen=True, slots=True)
class ExactDispatchRowTruth:
    """Backend-owned row truth used to derive distinct Dispatch and Combine bytes."""

    phase_key: "PhaseKey"
    src_rank: int
    realized_rows_by_destination: tuple[int, ...]
    dispatch_payload_bytes_by_destination: tuple[int, ...]
    combine_return_payload_bytes_by_expert: tuple[int, ...]
    dispatch_payload_spec_digest: StableDigest
    combine_payload_spec_digest: StableDigest
    descriptor_payload_bytes: int
    truth_digest: StableDigest

    def __post_init__(self) -> None:
        if not isinstance(self.phase_key, PhaseKey) or self.phase_key.phase_kind is not PhaseKind.DISPATCH:
            raise ValueError("ExactDispatchRowTruth requires Dispatch PhaseKey")
        _require_nonnegative("src_rank", self.src_rank)
        vectors = (
            self.realized_rows_by_destination,
            self.dispatch_payload_bytes_by_destination,
            self.combine_return_payload_bytes_by_expert,
        )
        if any(not isinstance(values, tuple) or not values for values in vectors):
            raise TypeError("row truth vectors must be non-empty tuples")
        if len({len(values) for values in vectors}) != 1:
            raise ValueError("row truth vectors must have equal world_size")
        for values in vectors:
            for value in values:
                _require_nonnegative("row truth value", value)
        for rows, dispatch_bytes, combine_bytes in zip(*vectors):
            if rows == 0 and (dispatch_bytes != 0 or combine_bytes != 0):
                raise ValueError("zero-row edge must be zero bytes in both phases")
            if rows > 0 and (dispatch_bytes <= 0 or combine_bytes <= 0):
                raise ValueError("nonzero-row edge must be positive bytes in both phases")
        for name in ("dispatch_payload_spec_digest", "combine_payload_spec_digest", "truth_digest"):
            _require_nonempty(name, getattr(self, name))
        _require_nonnegative("descriptor_payload_bytes", self.descriptor_payload_bytes)


@dataclass(frozen=True, slots=True)
class TransferBatch:
    batch_id: str
    batch_digest: StableDigest
    phase_key: "PhaseKey"
    task_ids: tuple[TaskId, ...]
    authority_stamp: AuthorityStamp
    link_class: LinkClass
    topology_digest: StableDigest
    compiled_at_ns: TimeNs

    def __post_init__(self) -> None:
        _require_nonempty("batch_id", self.batch_id)
        _require_nonempty("batch_digest", self.batch_digest)
        if not isinstance(self.phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        object.__setattr__(self, "task_ids", _validated_ids(self.task_ids, "task_ids"))
        if not self.task_ids:
            raise ValueError("task_ids must be non-empty")
        if not isinstance(self.authority_stamp, AuthorityStamp):
            raise TypeError("authority_stamp must be AuthorityStamp")
        if not isinstance(self.link_class, LinkClass):
            raise TypeError("link_class must be LinkClass")
        if self.link_class is LinkClass.LOCAL_ASSEMBLY:
            raise ValueError("local assembly must never be submitted to DataPlane")
        _require_nonempty("topology_digest", self.topology_digest)
        _require_nonnegative("compiled_at_ns", self.compiled_at_ns)

    @property
    def authority_token(self) -> str:
        return self.authority_stamp.phase_token

    @property
    def plan_id(self) -> PlanId:
        return self.authority_stamp.plan_id

    @property
    def phase_plan_epoch(self) -> int:
        return self.authority_stamp.phase_plan_epoch


@dataclass(frozen=True, slots=True)
class TransportSnapshot:
    snapshot_at_ns: TimeNs
    max_batch_tasks: int
    busy_src_ranks: tuple[int, ...]
    busy_dst_ranks: tuple[int, ...]
    busy_nic_ids: tuple[str, ...]
    busy_lane_ids: tuple[str, ...]
    available_lane_ids_by_link_class: tuple[tuple[LinkClass, tuple[str, ...]], ...]
    hardware_profile_digest: StableDigest
    topology_digest: StableDigest

    def __post_init__(self) -> None:
        _require_nonnegative("snapshot_at_ns", self.snapshot_at_ns)
        if not isinstance(self.max_batch_tasks, int) or isinstance(self.max_batch_tasks, bool) or self.max_batch_tasks <= 0:
            raise ValueError("max_batch_tasks must be positive")
        for name in ("busy_src_ranks", "busy_dst_ranks"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be tuple")
            for value in values:
                _require_nonnegative(name, value)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicates")
        for name in ("busy_nic_ids", "busy_lane_ids"):
            object.__setattr__(self, name, _validated_ids(getattr(self, name), name))
        if not isinstance(self.available_lane_ids_by_link_class, tuple):
            raise TypeError("available_lane_ids_by_link_class must be tuple")
        seen: set[LinkClass] = set()
        for link_class, lane_ids in self.available_lane_ids_by_link_class:
            if not isinstance(link_class, LinkClass):
                raise TypeError("available lane key must be LinkClass")
            if link_class in seen:
                raise ValueError("duplicate link class in available lanes")
            seen.add(link_class)
            _validated_ids(lane_ids, "available_lane_ids")
        _require_nonempty("hardware_profile_digest", self.hardware_profile_digest)
        _require_nonempty("topology_digest", self.topology_digest)


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    profile_id: str
    profile_digest: StableDigest
    profile_provenance: str
    performance_eligible: bool
    max_batch_tasks: int
    launch_delay_ns_by_link_class: tuple[tuple[LinkClass, int], ...]
    fixed_latency_ns_by_link_class: tuple[tuple[LinkClass, int], ...]
    bandwidth_bytes_per_second_by_link_class: tuple[tuple[LinkClass, int], ...]

    def __post_init__(self) -> None:
        _require_nonempty("profile_id", self.profile_id)
        _require_nonempty("profile_digest", self.profile_digest)
        _require_nonempty("profile_provenance", self.profile_provenance)
        if not isinstance(self.performance_eligible, bool):
            raise TypeError("performance_eligible must be bool")
        if not isinstance(self.max_batch_tasks, int) or isinstance(self.max_batch_tasks, bool) or self.max_batch_tasks <= 0:
            raise ValueError("max_batch_tasks must be positive")
        for name in (
            "launch_delay_ns_by_link_class",
            "fixed_latency_ns_by_link_class",
            "bandwidth_bytes_per_second_by_link_class",
        ):
            rows = getattr(self, name)
            if not isinstance(rows, tuple):
                raise TypeError(f"{name} must be tuple")
            seen: set[LinkClass] = set()
            for link_class, value in rows:
                if not isinstance(link_class, LinkClass) or link_class is LinkClass.LOCAL_ASSEMBLY:
                    raise TypeError(f"{name} keys must be non-local LinkClass")
                if link_class in seen:
                    raise ValueError(f"{name} contains duplicate link class")
                seen.add(link_class)
                if name == "bandwidth_bytes_per_second_by_link_class":
                    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                        raise ValueError("bandwidth_bytes_per_second must be positive")
                else:
                    _require_nonnegative(name, value)


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    receipt_id: str
    batch_id: str
    batch_digest: StableDigest
    phase_key: "PhaseKey"
    task_ids: tuple[TaskId, ...]
    authority_stamp: AuthorityStamp
    topology_digest: StableDigest
    commit_time_ns: TimeNs
    resource_reservation_digest: StableDigest
    transport_snapshot_digest: StableDigest

    def __post_init__(self) -> None:
        for name in (
            "receipt_id", "batch_id", "batch_digest", "topology_digest",
            "resource_reservation_digest", "transport_snapshot_digest",
        ):
            _require_nonempty(name, getattr(self, name))
        if not isinstance(self.phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        object.__setattr__(self, "task_ids", _validated_ids(self.task_ids, "task_ids"))
        if not self.task_ids:
            raise ValueError("task_ids must be non-empty")
        if not isinstance(self.authority_stamp, AuthorityStamp):
            raise TypeError("authority_stamp must be AuthorityStamp")
        _require_nonnegative("commit_time_ns", self.commit_time_ns)

    @property
    def authority_token(self) -> str:
        return self.authority_stamp.phase_token

    @property
    def plan_id(self) -> PlanId:
        return self.authority_stamp.plan_id

    @property
    def phase_plan_epoch(self) -> int:
        return self.authority_stamp.phase_plan_epoch


@dataclass(frozen=True, slots=True)
class PhysicalTransferRecord:
    task_id: TaskId
    batch_id: str
    link_class: LinkClass
    lane_id: str
    committed_at_ns: TimeNs
    start_at_ns: TimeNs
    complete_at_ns: TimeNs
    payload_bytes: int

    def __post_init__(self) -> None:
        for name in ("task_id", "batch_id", "lane_id"):
            _require_nonempty(name, getattr(self, name))
        if not isinstance(self.link_class, LinkClass):
            raise TypeError("link_class must be LinkClass")
        for name in ("committed_at_ns", "start_at_ns", "complete_at_ns", "payload_bytes"):
            _require_nonnegative(name, getattr(self, name))
        if self.start_at_ns < self.committed_at_ns or self.complete_at_ns < self.start_at_ns:
            raise ValueError("physical transfer timestamps are not monotonic")


@dataclass(frozen=True, slots=True)
class RowBroadcastRequest:
    phase_key: "PhaseKey"
    src_rank: int
    realized_rows_by_destination: tuple[int, ...]
    payload_bytes_by_destination: tuple[int, ...]
    payload_spec_digest: StableDigest
    descriptor_digest: StableDigest
    published_at_ns: TimeNs
    descriptor_payload_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase_key, PhaseKey) or self.phase_key.phase_kind is not PhaseKind.DISPATCH:
            raise ValueError("RowBroadcastRequest is Dispatch-only")
        _require_nonnegative("src_rank", self.src_rank)
        for name in ("realized_rows_by_destination", "payload_bytes_by_destination"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise TypeError(f"{name} must be a non-empty tuple")
            for value in values:
                _require_nonnegative(name, value)
        if len(self.realized_rows_by_destination) != len(self.payload_bytes_by_destination):
            raise ValueError("broadcast rows and payload bytes must have equal destination count")
        for name in ("payload_spec_digest", "descriptor_digest"):
            _require_nonempty(name, getattr(self, name))
        _require_nonnegative("published_at_ns", self.published_at_ns)
        _require_nonnegative("descriptor_payload_bytes", self.descriptor_payload_bytes)


@dataclass(frozen=True, slots=True)
class ControlPlaneDelivery:
    request_digest: StableDigest
    phase_key: "PhaseKey"
    src_rank: int
    delivery_start_ns: TimeNs
    delivered_at_ns: TimeNs
    control_channel_id: str

    def __post_init__(self) -> None:
        _require_nonempty("request_digest", self.request_digest)
        if not isinstance(self.phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        _require_nonnegative("src_rank", self.src_rank)
        _require_nonnegative("delivery_start_ns", self.delivery_start_ns)
        _require_nonnegative("delivered_at_ns", self.delivered_at_ns)
        if self.delivered_at_ns < self.delivery_start_ns:
            raise ValueError("delivered_at_ns precedes delivery_start_ns")
        _require_nonempty("control_channel_id", self.control_channel_id)


@dataclass(frozen=True, slots=True)
class TransferStarted:
    task_id: TaskId
    batch_id: str
    start_at_ns: TimeNs
    physical_record_digest: StableDigest

    def __post_init__(self) -> None:
        _require_nonempty("task_id", self.task_id)
        _require_nonempty("batch_id", self.batch_id)
        _require_nonnegative("start_at_ns", self.start_at_ns)
        _require_nonempty("physical_record_digest", self.physical_record_digest)


@dataclass(frozen=True, slots=True)
class TransferCompleted:
    task_id: TaskId
    batch_id: str
    complete_at_ns: TimeNs
    payload_bytes: int
    physical_record_digest: StableDigest

    def __post_init__(self) -> None:
        _require_nonempty("task_id", self.task_id)
        _require_nonempty("batch_id", self.batch_id)
        _require_nonnegative("complete_at_ns", self.complete_at_ns)
        if not isinstance(self.payload_bytes, int) or isinstance(self.payload_bytes, bool) or self.payload_bytes <= 0:
            raise ValueError("payload_bytes must be positive")
        _require_nonempty("physical_record_digest", self.physical_record_digest)


class KernelPhase(IntEnum):
    COMPLETION_COLLECTION = 1
    AUTHORITATIVE_STATE_UPDATES = 2
    DESCRIPTOR_OBSERVATION_DELIVERY = 3
    BACKEND_RECEIVER_CLOSURE_RELEASE = 4
    THREE_LINE_JOB_TRANSITIONS = 5
    PLAN_ACTIVATION_SUPERSESSION = 6
    EXECUTION_STABILIZATION_SUBMIT = 7
    DEADLOCK_PROGRESS_CHECK = 8


@dataclass(frozen=True, slots=True, order=True)
class WindowKey:
    run_id: str
    sample_id: str
    window_index: int

    def __post_init__(self) -> None:
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("sample_id", self.sample_id)
        _require_nonnegative("window_index", self.window_index)


@dataclass(frozen=True, slots=True, order=True)
class PhaseKey:
    run_id: str
    sample_id: str
    layer_index: int
    phase_kind: PhaseKind

    def __post_init__(self) -> None:
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("sample_id", self.sample_id)
        _require_nonnegative("layer_index", self.layer_index)
        if not isinstance(self.phase_kind, PhaseKind):
            raise TypeError("phase_kind must be PhaseKind")


@dataclass(frozen=True, slots=True, order=True)
class EdgeKey:
    phase_key: PhaseKey
    src_rank: int
    dst_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        _require_nonnegative("src_rank", self.src_rank)
        _require_nonnegative("dst_rank", self.dst_rank)


@dataclass(frozen=True, slots=True)
class ReceiveExpectation:
    edge_key: EdgeKey
    phase_key: PhaseKey
    src_rank: int
    dst_rank: int
    total_expected_payload_bytes: int
    expectation_digest: StableDigest
    origin: ExpectationOrigin
    created_at_ns: TimeNs
    zero_edge: bool
    descriptor_digest_or_none: StableDigest | None

    def __post_init__(self) -> None:
        if not isinstance(self.edge_key, EdgeKey):
            raise TypeError("edge_key must be EdgeKey")
        if not isinstance(self.phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        if self.edge_key.phase_key != self.phase_key:
            raise ValueError("edge_key.phase_key does not match phase_key")
        if self.edge_key.src_rank != self.src_rank or self.edge_key.dst_rank != self.dst_rank:
            raise ValueError("edge_key ranks do not match expectation ranks")
        _require_nonnegative("total_expected_payload_bytes", self.total_expected_payload_bytes)
        _require_nonnegative("created_at_ns", self.created_at_ns)
        _require_nonempty("expectation_digest", self.expectation_digest)
        if not isinstance(self.origin, ExpectationOrigin):
            raise TypeError("origin must be ExpectationOrigin")
        if self.zero_edge != (self.total_expected_payload_bytes == 0):
            raise ValueError("zero_edge must exactly match zero payload bytes")
        if self.origin is ExpectationOrigin.DISPATCH_DESCRIPTOR:
            if self.phase_key.phase_kind is not PhaseKind.DISPATCH:
                raise ValueError("Dispatch expectation origin requires Dispatch phase")
            if not self.descriptor_digest_or_none:
                raise ValueError("Dispatch expectation requires descriptor digest")
            _require_nonempty(
                "descriptor_digest_or_none", self.descriptor_digest_or_none
            )
        else:
            if self.phase_key.phase_kind is not PhaseKind.COMBINE:
                raise ValueError("Combine expectation origin requires Combine phase")
            if self.descriptor_digest_or_none is not None:
                raise ValueError("Combine expectation descriptor digest must be None")


@dataclass(frozen=True, slots=True)
class CanonicalTransferTask:
    task_id: TaskId
    edge_key: EdgeKey
    phase_key: PhaseKey
    src_rank: int
    dst_rank: int
    chunk_index: int
    byte_offset: int
    payload_bytes: int
    expectation_digest: StableDigest
    taskization_digest: StableDigest
    registered_at_ns: TimeNs

    def __post_init__(self) -> None:
        if not isinstance(self.edge_key, EdgeKey):
            raise TypeError("edge_key must be EdgeKey")
        if not isinstance(self.phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        _require_nonempty("task_id", self.task_id)
        _require_nonempty("expectation_digest", self.expectation_digest)
        _require_nonempty("taskization_digest", self.taskization_digest)
        if self.edge_key.phase_key != self.phase_key:
            raise ValueError("edge_key.phase_key does not match phase_key")
        if self.edge_key.src_rank != self.src_rank or self.edge_key.dst_rank != self.dst_rank:
            raise ValueError("edge_key ranks do not match task ranks")
        _require_nonnegative("chunk_index", self.chunk_index)
        _require_nonnegative("byte_offset", self.byte_offset)
        if not isinstance(self.payload_bytes, int) or isinstance(self.payload_bytes, bool) or self.payload_bytes <= 0:
            raise ValueError("payload_bytes must be a positive int; zero edges create no task")
        _require_nonnegative("registered_at_ns", self.registered_at_ns)


@dataclass(frozen=True, slots=True)
class ReceivePermit:
    permit_id: str
    task_id: TaskId
    edge_key: EdgeKey
    chunk_index: int
    byte_offset: int
    task_bytes: int
    credit_reservation_id: str
    expectation_digest: StableDigest
    descriptor_digest_or_none: StableDigest | None
    posted_at_ns: TimeNs

    def __post_init__(self) -> None:
        if not isinstance(self.edge_key, EdgeKey):
            raise TypeError("edge_key must be EdgeKey")
        for name, value in (
            ("permit_id", self.permit_id),
            ("task_id", self.task_id),
            ("credit_reservation_id", self.credit_reservation_id),
            ("expectation_digest", self.expectation_digest),
        ):
            _require_nonempty(name, value)
        _require_nonnegative("chunk_index", self.chunk_index)
        _require_nonnegative("byte_offset", self.byte_offset)
        if not isinstance(self.task_bytes, int) or isinstance(self.task_bytes, bool) or self.task_bytes <= 0:
            raise ValueError("task_bytes must be a positive int")
        _require_nonnegative("posted_at_ns", self.posted_at_ns)
        if self.edge_key.phase_key.phase_kind is PhaseKind.DISPATCH:
            if not self.descriptor_digest_or_none:
                raise ValueError("Dispatch permit requires descriptor digest")
            _require_nonempty(
                "descriptor_digest_or_none", self.descriptor_digest_or_none
            )
        elif self.descriptor_digest_or_none is not None:
            raise ValueError("Combine permit descriptor digest must be None")


@dataclass(frozen=True, slots=True)
class PhaseExecutionRecord:
    phase_key: PhaseKey
    canonical_task_ids: tuple[TaskId, ...]
    task_catalogue_digest: StableDigest
    active_plan_id: PlanId | None
    phase_plan_epoch: int
    committed_task_ids: tuple[TaskId, ...]
    running_task_ids: tuple[TaskId, ...]
    completed_task_ids: tuple[TaskId, ...]
    registered_window_keys: tuple[WindowKey, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        if self.active_plan_id is not None:
            _require_nonempty("active_plan_id", self.active_plan_id)
        object.__setattr__(self, "canonical_task_ids", _validated_ids(self.canonical_task_ids, "canonical_task_ids"))
        object.__setattr__(self, "committed_task_ids", _validated_ids(self.committed_task_ids, "committed_task_ids"))
        object.__setattr__(self, "running_task_ids", _validated_ids(self.running_task_ids, "running_task_ids"))
        object.__setattr__(self, "completed_task_ids", _validated_ids(self.completed_task_ids, "completed_task_ids"))
        if not isinstance(self.registered_window_keys, tuple):
            raise TypeError("registered_window_keys must be a tuple")
        if not all(
            isinstance(window_key, WindowKey)
            for window_key in self.registered_window_keys
        ):
            raise TypeError("registered_window_keys must contain WindowKey values")
        if len(set(self.registered_window_keys)) != len(self.registered_window_keys):
            raise ValueError("registered_window_keys contains duplicates")
        _require_nonempty("task_catalogue_digest", self.task_catalogue_digest)
        _require_nonnegative("phase_plan_epoch", self.phase_plan_epoch)
        catalogue = set(self.canonical_task_ids)
        for name in ("committed_task_ids", "running_task_ids", "completed_task_ids"):
            if not set(getattr(self, name)).issubset(catalogue):
                raise ValueError(f"{name} must be a subset of canonical_task_ids")
        state_sets = {
            "committed": set(self.committed_task_ids),
            "running": set(self.running_task_ids),
            "completed": set(self.completed_task_ids),
        }
        state_names = tuple(state_sets)
        for index, left_name in enumerate(state_names):
            for right_name in state_names[index + 1 :]:
                if state_sets[left_name] & state_sets[right_name]:
                    raise ValueError(
                        f"{left_name} and {right_name} task sets overlap"
                    )


@dataclass(frozen=True, slots=True)
class PlanVersion:
    plan_id: PlanId
    window_key: WindowKey
    version: int
    status: PlanStatus
    supersedes_plan_ids: tuple[PlanId, ...]
    commit_index: int | None
    committed_task_ids: tuple[TaskId, ...]
    remaining_task_ids: tuple[TaskId, ...]
    created_at_ns: TimeNs
    activated_at_ns: TimeNs | None
    completed_at_ns: TimeNs | None
    plan_digest: StableDigest

    def __post_init__(self) -> None:
        if not isinstance(self.window_key, WindowKey):
            raise TypeError("window_key must be WindowKey")
        if not isinstance(self.status, PlanStatus):
            raise TypeError("status must be PlanStatus")
        _require_nonempty("plan_id", self.plan_id)
        _require_nonempty("plan_digest", self.plan_digest)
        _require_nonnegative("version", self.version)
        _require_nonnegative("created_at_ns", self.created_at_ns)
        if self.commit_index is not None:
            _require_nonnegative("commit_index", self.commit_index)
        if self.activated_at_ns is not None:
            _require_nonnegative("activated_at_ns", self.activated_at_ns)
            if self.activated_at_ns < self.created_at_ns:
                raise ValueError("activated_at_ns precedes created_at_ns")
        if self.completed_at_ns is not None:
            _require_nonnegative("completed_at_ns", self.completed_at_ns)
            if self.completed_at_ns < self.created_at_ns:
                raise ValueError("completed_at_ns precedes created_at_ns")
            if (
                self.activated_at_ns is not None
                and self.completed_at_ns < self.activated_at_ns
            ):
                raise ValueError("completed_at_ns precedes activated_at_ns")
        object.__setattr__(self, "supersedes_plan_ids", _validated_ids(self.supersedes_plan_ids, "supersedes_plan_ids"))
        object.__setattr__(self, "committed_task_ids", _validated_ids(self.committed_task_ids, "committed_task_ids"))
        object.__setattr__(self, "remaining_task_ids", _validated_ids(self.remaining_task_ids, "remaining_task_ids"))
        if set(self.committed_task_ids) & set(self.remaining_task_ids):
            raise ValueError("committed and remaining task sets overlap")


@dataclass(frozen=True, slots=True, order=True)
class SimulationEvent:
    time_ns: TimeNs
    round_index: int
    phase_priority: KernelPhase
    stable_event_id: str
    producer: str
    event_type: str
    subject_id: str
    ordinal: int
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative("time_ns", self.time_ns)
        _require_nonnegative("round_index", self.round_index)
        _require_nonnegative("ordinal", self.ordinal)
        for name, value in (
            ("stable_event_id", self.stable_event_id),
            ("producer", self.producer),
            ("event_type", self.event_type),
        ):
            _require_nonempty(name, value)
        if not isinstance(self.phase_priority, KernelPhase):
            raise TypeError("phase_priority must be KernelPhase")
        if not isinstance(self.subject_id, str):
            raise TypeError("subject_id must be str")
        if not isinstance(self.attributes, tuple):
            raise TypeError("attributes must be a tuple")
        keys: list[str] = []
        for item in self.attributes:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("attributes entries must be (str, str) tuples")
            key, value = item
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("attributes entries must contain strings")
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise ValueError("attributes contain duplicate keys")
        object.__setattr__(self, "attributes", tuple(sorted(self.attributes)))

    @property
    def sort_key(self) -> tuple[int, int, int, str]:
        return (self.time_ns, self.round_index, int(self.phase_priority), self.stable_event_id)


@dataclass(frozen=True, slots=True)
class GoldenTimelineRow:
    timeline_index: int
    time_ns: TimeNs
    round_index: int
    phase_priority: KernelPhase
    stable_event_id: str
    producer: str
    event_type: str
    subject_id: str
    outcome: str
    details_digest: StableDigest

    def __post_init__(self) -> None:
        _require_nonnegative("timeline_index", self.timeline_index)
        _require_nonnegative("time_ns", self.time_ns)
        _require_nonnegative("round_index", self.round_index)
        if not isinstance(self.phase_priority, KernelPhase):
            raise TypeError("phase_priority must be KernelPhase")
        if not isinstance(self.subject_id, str):
            raise TypeError("subject_id must be str")
        for name, value in (
            ("stable_event_id", self.stable_event_id),
            ("producer", self.producer),
            ("event_type", self.event_type),
            ("outcome", self.outcome),
            ("details_digest", self.details_digest),
        ):
            _require_nonempty(name, value)
