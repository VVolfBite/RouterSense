"""Backend-owned mutable state.

None of these records duplicate shared contract objects.  They are private
bookkeeping for the backend's unique ownership of receiver, buffer, closure and release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReceiverJobStatus(str, Enum):
    WAITING_INPUTS = "WAITING_INPUTS"
    ELIGIBLE = "ELIGIBLE"
    POSTING = "POSTING"
    POSTED = "POSTED"
    TRANSFER_COMPLETED = "TRANSFER_COMPLETED"
    DRAINING = "DRAINING"
    ASSEMBLED = "ASSEMBLED"


class RankState(str, Enum):
    WAIT_DISPATCH = "WAIT_DISPATCH"
    EXPERT_COMPUTE = "EXPERT_COMPUTE"
    WAIT_COMBINE = "WAIT_COMBINE"
    WAIT_PHASE_BARRIER = "WAIT_PHASE_BARRIER"
    POST_COMBINE_LOCAL_PATH = "POST_COMBINE_LOCAL_PATH"
    ROUTER_AND_PACK = "ROUTER_AND_PACK"
    DONE = "DONE"


@dataclass(slots=True)
class EdgeState:
    edge_key: Any
    edge_stable_key: str
    phase_key: Any
    phase_stable_key: str
    phase_kind: str
    src_rank: int
    dst_rank: int
    expected_bytes: int
    expectation_digest: str
    origin: str
    expectation_available_at_ns: int
    zero_edge: bool
    descriptor_digest_or_none: str | None
    expectation_object: Any
    task_ids: list[Any] = field(default_factory=list)
    catalogue_validated: bool = False
    assembled_bytes: int = 0
    data_complete_at_ns: int | None = None


@dataclass(slots=True)
class TaskStateRecord:
    task_object: Any
    task_id: Any
    task_stable_key: str
    edge_key: Any
    edge_stable_key: str
    phase_key: Any
    phase_stable_key: str
    src_rank: int
    dst_rank: int
    chunk_index: int
    byte_offset: int
    payload_bytes: int
    registered_at_ns: int
    requested_at_ns: int | None = None
    eligible_at_ns: int | None = None
    status: ReceiverJobStatus = ReceiverJobStatus.WAITING_INPUTS
    reservation_id: str | None = None
    permit_object: Any | None = None
    receiver_start_ns: int | None = None
    receive_posted_at_ns: int | None = None
    transfer_complete_at_ns: int | None = None
    drain_start_ns: int | None = None
    drain_finish_ns: int | None = None
    buffer_stall_started_ns: int | None = None
    buffer_stall_ns: int = 0
    drain_queue_wait_ns: int = 0

    def fifo_key(self) -> tuple[int, int, str, int, int, str]:
        assert self.eligible_at_ns is not None
        assert self.requested_at_ns is not None
        return (
            self.eligible_at_ns,
            self.requested_at_ns,
            self.phase_stable_key,
            self.src_rank,
            self.chunk_index,
            self.task_stable_key,
        )


@dataclass(slots=True)
class PhaseDestinationMetrics:
    """Per-phase/per-destination receiver accounting.

    This record mirrors receiver-owned state.  It is intentionally
    separate from the fixed cross-phase staging pool so reports can attribute
    waits and memory peaks without changing capacity semantics.
    """

    current_staging_bytes: int = 0
    current_final_assembly_bytes: int = 0
    peak_staging_bytes: int = 0
    peak_final_assembly_bytes: int = 0
    peak_total_receiver_bytes: int = 0
    receiver_buffer_stall_ns: int = 0
    receiver_posting_service_ns: int = 0
    receiver_posting_queue_wait_ns: int = 0
    receiver_drain_queue_wait_ns: int = 0
    receiver_drain_service_ns: int = 0

    def update_peaks(self) -> None:
        self.peak_staging_bytes = max(
            self.peak_staging_bytes, self.current_staging_bytes
        )
        self.peak_final_assembly_bytes = max(
            self.peak_final_assembly_bytes, self.current_final_assembly_bytes
        )
        self.peak_total_receiver_bytes = max(
            self.peak_total_receiver_bytes,
            self.current_staging_bytes + self.current_final_assembly_bytes,
        )


@dataclass(slots=True)
class DestinationMemory:
    capacity_bytes: int | None
    reserved_bytes: int = 0
    used_bytes: int = 0
    final_assembly_bytes: int = 0
    peak_staging_bytes: int = 0
    peak_final_assembly_bytes: int = 0
    peak_total_receiver_bytes: int = 0
    receiver_buffer_stall_ns: int = 0
    receiver_posting_service_ns: int = 0
    receiver_posting_queue_wait_ns: int = 0
    receiver_drain_queue_wait_ns: int = 0
    receiver_drain_service_ns: int = 0

    @property
    def staging_bytes(self) -> int:
        return self.reserved_bytes + self.used_bytes

    @property
    def free_bytes(self) -> int | None:
        if self.capacity_bytes is None:
            return None
        return self.capacity_bytes - self.staging_bytes

    def update_peaks(self) -> None:
        staging = self.staging_bytes
        self.peak_staging_bytes = max(self.peak_staging_bytes, staging)
        self.peak_final_assembly_bytes = max(
            self.peak_final_assembly_bytes, self.final_assembly_bytes
        )
        self.peak_total_receiver_bytes = max(
            self.peak_total_receiver_bytes, staging + self.final_assembly_bytes
        )


@dataclass(slots=True)
class PostingServerState:
    available_at_ns: int = 0
    active_task_id: Any | None = None
    eligible_task_ids: set[Any] = field(default_factory=set)


@dataclass(slots=True)
class DrainLineState:
    available_at_ns: int = 0
    active_task_id: Any | None = None
    waiting_task_ids: set[Any] = field(default_factory=set)


@dataclass(slots=True)
class DispatchDestinationState:
    phase_key: Any
    phase_stable_key: str
    dst_rank: int
    descriptor_sources_delivered: dict[int, int] = field(default_factory=dict)
    descriptor_closure_at_ns: int | None = None
    model_thread_ready_at_ns: int | None = None
    all_inbound_assembled_at_ns: int | None = None
    postprocess_start_at_ns: int | None = None
    compute_ready_at_ns: int | None = None
    postprocess_scheduled: bool = False
    final_assembly_released: bool = False
    released: bool = False
    closure_wait_ns: int = 0


@dataclass(slots=True)
class CombineDestinationState:
    phase_key: Any
    phase_stable_key: str
    dst_rank: int
    data_ready_at_ns: int | None = None
    local_path_start_at_ns: int | None = None
    post_combine_complete_at_ns: int | None = None
    local_path_complete_at_ns: int | None = None
    post_combine_scheduled: bool = False
    local_path_scheduled: bool = False


@dataclass(frozen=True, slots=True)
class LocalPathSpec:
    combine_phase_key: Any
    next_dispatch_phase_key: Any | None
    rank_id: int
    combine_release_to_router_ready_ns: int
    router_and_pack_ns: int


@dataclass(frozen=True, slots=True)
class DispatchComputeSpec:
    dispatch_phase_key: Any
    next_combine_phase_key: Any
    rank_id: int
    dispatch_local_postprocess_ns: int
    dispatch_release_to_combine_source_ready_ns: int
