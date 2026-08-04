"""Receiver-decoupled posting, staging, drain and final assembly service."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from rs_sim.backend.core.errors import (
    BackendContractError,
    CapacityConfigurationError,
    DuplicateRegistrationError,
    IllegalTransitionError,
    UnknownObjectError,
)
from rs_sim.backend.core.internal import (
    DestinationMemory,
    DrainLineState,
    EdgeState,
    PhaseDestinationMetrics,
    PostingServerState,
    ReceiverJobStatus,
    TaskStateRecord,
)
from rs_sim.backend.observability.metrics import ReceiverMetricsSnapshot, snapshot_memory
from rs_sim.backend.core.ports import (
    BackendObserverPort,
    CostModel,
    KernelPort,
    PermitFactory,
    PhaseSemantics,
    SharedObjectAdapter,
)
from rs_sim.backend.core.util import (
    require_nonnegative_int,
    require_positive_int,
    require_time_ns,
    stable_semantic_event_id,
)

BACKEND_PHASE_PRIORITY = 4
_EVENT_RECEIVER_POST_COMPLETE = "BACKEND_RECEIVER_POST_COMPLETE"
_EVENT_RECEIVER_DRAIN_FINISH = "BACKEND_RECEIVER_DRAIN_FINISH"
_EVENT_LOCAL_ASSEMBLY_FINISH = "BACKEND_LOCAL_ASSEMBLY_FINISH"


class ReceiverService:
    """Receiver service shared across all active phases per destination."""

    def __init__(
        self,
        *,
        world_size: int,
        staging_capacity_bytes_by_rank: Mapping[int, int | None],
        kernel: KernelPort,
        observer: BackendObserverPort,
        adapter: SharedObjectAdapter,
        phase_semantics: PhaseSemantics,
        permit_factory: PermitFactory,
        cost_model: CostModel,
        local_assembly_cost_ns: int = 0,
        allow_legacy_local_network_tasks_for_unit_tests: bool = False,
    ) -> None:
        if not isinstance(world_size, int) or world_size <= 0:
            raise BackendContractError("world_size must be a positive int")
        self.world_size = world_size
        self.kernel = kernel
        self.observer = observer
        self.adapter = adapter
        self.phase_semantics = phase_semantics
        self.permit_factory = permit_factory
        self.cost_model = cost_model
        self.local_assembly_cost_ns = require_nonnegative_int(
            local_assembly_cost_ns, field="local_assembly_cost_ns"
        )
        self._local_assembly_scheduled: set[str] = set()
        self.allow_legacy_local_network_tasks_for_unit_tests = bool(
            allow_legacy_local_network_tasks_for_unit_tests
        )

        expected_ranks = set(range(world_size))
        if set(staging_capacity_bytes_by_rank) != expected_ranks:
            raise BackendContractError(
                "staging capacity must be fixed for every destination rank"
            )
        self.memory_by_rank: dict[int, DestinationMemory] = {}
        for rank in range(world_size):
            capacity = staging_capacity_bytes_by_rank[rank]
            if capacity is not None:
                capacity = require_positive_int(
                    capacity, field=f"staging_capacity_bytes_by_rank[{rank}]"
                )
            self.memory_by_rank[rank] = DestinationMemory(capacity_bytes=capacity)

        self.posting_by_rank = {
            rank: PostingServerState() for rank in range(world_size)
        }
        self.drain_by_rank = {rank: DrainLineState() for rank in range(world_size)}

        self.edges_by_key: dict[str, EdgeState] = {}
        self._edges_by_phase_dst: dict[tuple[str, int], list[EdgeState]] = defaultdict(list)
        self._assembled_completion_cache: dict[tuple[str, int], tuple[bool, int | None]] = {}
        self.tasks_by_key: dict[str, TaskStateRecord] = {}
        self._task_key_by_external_id: dict[str, str] = {}
        self._pending_task_keys_by_edge: dict[str, list[str]] = defaultdict(list)
        self._catalogue_declared_edges: set[str] = set()
        self._source_payload_ready_at: dict[tuple[str, int], int] = {}
        self._final_assembly_by_phase_dst: dict[tuple[str, int], int] = defaultdict(int)
        self._released_final_assembly: set[tuple[str, int]] = set()
        self._sealed_combine_destinations: dict[tuple[str, int], int] = {}
        self._finalized_expectation_closure_by_phase: dict[str, str] = {}
        self._phase_metrics_by_phase_dst: dict[
            tuple[str, int], PhaseDestinationMetrics
        ] = {}
        # Exact receiver-memory observations at every authoritative mutation.
        # Rows store the total cross-phase memory after the mutation so a P12
        # window can report overlap between P1 and P2 instead of taking the
        # maximum of two phase-local peaks.
        self._memory_journal: list[tuple[int, str, int, int, int, int]] = []

    # ------------------------------------------------------------------
    # Expectation and canonical catalogue registration
    # ------------------------------------------------------------------
    def register_expectation(
        self,
        expectation: Any,
        *,
        descriptor_digest_or_none: str | None,
    ) -> None:
        edge_key = self.adapter.get(expectation, "edge_key")
        phase_key = self.adapter.get(expectation, "phase_key")
        src_rank = require_nonnegative_int(
            self.adapter.get(expectation, "src_rank"), field="expectation.src_rank"
        )
        dst_rank = require_nonnegative_int(
            self.adapter.get(expectation, "dst_rank"), field="expectation.dst_rank"
        )
        if src_rank >= self.world_size or dst_rank >= self.world_size:
            raise BackendContractError("expectation rank is outside world_size")
        expected_bytes = require_nonnegative_int(
            self.adapter.get(expectation, "total_expected_payload_bytes"),
            field="expectation.total_expected_payload_bytes",
        )
        expectation_digest = str(
            self.adapter.get(expectation, "expectation_digest")
        )
        origin = str(self.adapter.get(expectation, "origin"))
        created_at_ns = require_time_ns(
            self.adapter.get(expectation, "created_at_ns"),
            field="expectation.created_at_ns",
        )
        zero_edge = bool(self.adapter.get(expectation, "zero_edge"))
        if zero_edge != (expected_bytes == 0):
            raise BackendContractError(
                "zero_edge must be true exactly when expected payload bytes are zero"
            )

        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        edge_stable_key = self.adapter.stable_key(edge_key)
        phase_kind = self.phase_semantics.phase_kind(phase_key)
        self._phase_metrics(phase_stable_key=phase_stable_key, dst_rank=dst_rank)
        object_descriptor_digest = None
        try:
            object_descriptor_digest = self.adapter.get(
                expectation, "descriptor_digest_or_none"
            )
        except BackendContractError:
            # Compatibility with pre-shared-schema fixture objects; production shared-schema objects carry it.
            object_descriptor_digest = descriptor_digest_or_none
        if object_descriptor_digest != descriptor_digest_or_none:
            raise BackendContractError(
                "expectation descriptor digest disagrees with registration argument"
            )
        if phase_kind == "DISPATCH" and not descriptor_digest_or_none:
            raise BackendContractError(
                "Dispatch expectation requires its delivered descriptor digest"
            )
        if phase_kind == "COMBINE" and descriptor_digest_or_none is not None:
            raise BackendContractError(
                "Combine expectation descriptor digest must be None"
            )

        candidate = EdgeState(
            edge_key=edge_key,
            edge_stable_key=edge_stable_key,
            phase_key=phase_key,
            phase_stable_key=phase_stable_key,
            phase_kind=phase_kind,
            src_rank=src_rank,
            dst_rank=dst_rank,
            expected_bytes=expected_bytes,
            expectation_digest=expectation_digest,
            origin=origin,
            expectation_available_at_ns=created_at_ns,
            zero_edge=zero_edge,
            descriptor_digest_or_none=descriptor_digest_or_none,
            expectation_object=expectation,
        )
        existing = self.edges_by_key.get(edge_stable_key)
        if existing is not None:
            immutable_old = (
                existing.phase_stable_key,
                existing.src_rank,
                existing.dst_rank,
                existing.expected_bytes,
                existing.expectation_digest,
                existing.expectation_available_at_ns,
                existing.zero_edge,
                existing.descriptor_digest_or_none,
            )
            immutable_new = (
                candidate.phase_stable_key,
                candidate.src_rank,
                candidate.dst_rank,
                candidate.expected_bytes,
                candidate.expectation_digest,
                candidate.expectation_available_at_ns,
                candidate.zero_edge,
                candidate.descriptor_digest_or_none,
            )
            if immutable_old != immutable_new:
                raise DuplicateRegistrationError(
                    f"edge {edge_stable_key} expectation changed after registration"
                )
            return

        if phase_stable_key in self._finalized_expectation_closure_by_phase:
            raise IllegalTransitionError(
                "late expectation registration after finalized phase closure"
            )
        self.edges_by_key[edge_stable_key] = candidate
        phase_dst_key = (phase_stable_key, dst_rank)
        indexed_edges = self._edges_by_phase_dst[phase_dst_key]
        indexed_edges.append(candidate)
        indexed_edges.sort(key=lambda item: (item.src_rank, item.edge_stable_key))
        self._assembled_completion_cache.pop(phase_dst_key, None)
        if zero_edge and self._pending_task_keys_by_edge.get(edge_stable_key):
            raise BackendContractError("zero edge cannot have canonical tasks")
        if src_rank == dst_rank and not self.allow_legacy_local_network_tasks_for_unit_tests:
            # Diagonal edges are local assembly truth, not network work.
            # No canonical network task or ReceivePermit is permitted.
            if self._pending_task_keys_by_edge.get(edge_stable_key):
                raise BackendContractError("local diagonal edge cannot have DataPlane tasks")
            candidate.catalogue_validated = True
        elif edge_stable_key in self._catalogue_declared_edges:
            self._validate_catalogue_for_edge(candidate)
        self._refresh_edge_jobs(candidate, now_ns=created_at_ns)
        self._refresh_local_assembly(candidate, now_ns=created_at_ns)
        self.observer.emit(
            kind="RECEIVE_EXPECTATION_AVAILABLE",
            at_ns=created_at_ns,
            payload={
                "edge_key": edge_key,
                "phase_key": phase_key,
                "src_rank": src_rank,
                "dst_rank": dst_rank,
                "zero_edge": zero_edge,
                "expected_bytes": expected_bytes,
                "expectation": expectation,
            },
        )

    def finalize_expectation_closure(
        self, *, phase_key: Any, closure_digest: str
    ) -> None:
        """Freeze the expectation set for one phase after Backend closure.

        Exact idempotent expectation replays remain accepted by
        ``register_expectation``.  Any new or conflicting edge is rejected.
        """

        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        closure_digest = str(closure_digest)
        if not closure_digest:
            raise BackendContractError("closure_digest must be non-empty")
        existing = self._finalized_expectation_closure_by_phase.get(
            phase_stable_key
        )
        if existing is not None:
            if existing != closure_digest:
                raise IllegalTransitionError(
                    "same phase finalized with a conflicting closure digest"
                )
            return
        self._finalized_expectation_closure_by_phase[
            phase_stable_key
        ] = closure_digest

    def expectation_closure_digest(self, *, phase_key: Any) -> str | None:
        return self._finalized_expectation_closure_by_phase.get(
            self.phase_semantics.phase_sort_key(phase_key)
        )

    def register_task_catalogue(self, tasks: Sequence[Any]) -> None:
        """Register complete immutable task catalogues for the represented edges."""
        grouped: dict[str, list[str]] = defaultdict(list)
        for task in tasks:
            task_id = self.adapter.get(task, "task_id")
            task_stable_key = self.adapter.stable_key(task_id)
            if task_stable_key in self.tasks_by_key:
                existing = self.tasks_by_key[task_stable_key]
                if self.adapter.stable_key(existing.task_object) != self.adapter.stable_key(
                    task
                ):
                    raise DuplicateRegistrationError(
                        f"task {task_stable_key} changed after registration"
                    )
                grouped[existing.edge_stable_key].append(task_stable_key)
                continue

            edge_key = self.adapter.get(task, "edge_key")
            phase_key = self.adapter.get(task, "phase_key")
            src_rank = require_nonnegative_int(
                self.adapter.get(task, "src_rank"), field="task.src_rank"
            )
            dst_rank = require_nonnegative_int(
                self.adapter.get(task, "dst_rank"), field="task.dst_rank"
            )
            chunk_index = require_nonnegative_int(
                self.adapter.get(task, "chunk_index"), field="task.chunk_index"
            )
            byte_offset = require_nonnegative_int(
                self.adapter.get(task, "byte_offset"), field="task.byte_offset"
            )
            payload_bytes = require_positive_int(
                self.adapter.get(task, "payload_bytes"), field="task.payload_bytes"
            )
            registered_at_ns = require_time_ns(
                self.adapter.get(task, "registered_at_ns"),
                field="task.registered_at_ns",
            )
            if src_rank >= self.world_size or dst_rank >= self.world_size:
                raise BackendContractError("task rank is outside world_size")
            edge_stable_key = self.adapter.stable_key(edge_key)
            phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
            record = TaskStateRecord(
                task_object=task,
                task_id=task_id,
                task_stable_key=task_stable_key,
                edge_key=edge_key,
                edge_stable_key=edge_stable_key,
                phase_key=phase_key,
                phase_stable_key=phase_stable_key,
                src_rank=src_rank,
                dst_rank=dst_rank,
                chunk_index=chunk_index,
                byte_offset=byte_offset,
                payload_bytes=payload_bytes,
                registered_at_ns=registered_at_ns,
            )
            self.tasks_by_key[task_stable_key] = record
            self.observer.emit(
                kind="CANONICAL_TASK_REGISTERED",
                at_ns=registered_at_ns,
                payload={
                    "task_id": task_id,
                    "edge_key": edge_key,
                    "phase_key": phase_key,
                    "src_rank": src_rank,
                    "dst_rank": dst_rank,
                    "chunk_index": chunk_index,
                    "byte_offset": byte_offset,
                    "payload_bytes": payload_bytes,
                },
            )
            external_id_key = self.adapter.stable_key(task_id)
            self._task_key_by_external_id[external_id_key] = task_stable_key
            self._pending_task_keys_by_edge[edge_stable_key].append(task_stable_key)
            grouped[edge_stable_key].append(task_stable_key)

        for edge_stable_key in sorted(grouped):
            if edge_stable_key in self._catalogue_declared_edges:
                # Exact idempotent replay is accepted, extension is not.
                current = set(self._pending_task_keys_by_edge[edge_stable_key])
                incoming = set(grouped[edge_stable_key])
                if incoming != current:
                    raise DuplicateRegistrationError(
                        f"canonical catalogue for edge {edge_stable_key} was extended"
                    )
            self._catalogue_declared_edges.add(edge_stable_key)
            edge = self.edges_by_key.get(edge_stable_key)
            if edge is not None:
                self._validate_catalogue_for_edge(edge)
                self._refresh_edge_jobs(edge, now_ns=max(
                    self.tasks_by_key[key].registered_at_ns
                    for key in self._pending_task_keys_by_edge[edge_stable_key]
                ))

    def seal_combine_expectations(
        self, *, phase_key: Any, dst_rank: int, at_ns: int
    ) -> None:
        at_ns = require_time_ns(at_ns, field="combine_expectation_closure.at_ns")
        if not isinstance(dst_rank, int) or isinstance(dst_rank, bool):
            raise BackendContractError("combine expectation destination must be an int")
        if dst_rank < 0 or dst_rank >= self.world_size:
            raise BackendContractError("combine expectation destination is outside world_size")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        if self.phase_semantics.phase_kind(phase_key) != "COMBINE":
            raise BackendContractError("only Combine expectations may be sealed here")
        edges = self.expected_edges_for_destination(
            phase_key=phase_key, dst_rank=dst_rank
        )
        if len(edges) != self.world_size:
            raise BackendContractError(
                "Combine expectation closure requires one edge per source rank"
            )
        key = (phase_stable_key, dst_rank)
        existing = self._sealed_combine_destinations.get(key)
        if existing is not None:
            if existing != at_ns:
                raise DuplicateRegistrationError(
                    "Combine expectation closure timestamp changed"
                )
            return
        self._sealed_combine_destinations[key] = at_ns
        self.observer.emit(
            kind="COMBINE_EXPECTATION_CLOSED",
            at_ns=at_ns,
            payload={"phase_key": phase_key, "dst_rank": dst_rank},
        )

    def _validate_catalogue_for_edge(self, edge: EdgeState) -> None:
        task_keys = self._pending_task_keys_by_edge.get(edge.edge_stable_key, [])
        if edge.zero_edge:
            if task_keys:
                raise BackendContractError("zero edge cannot have tasks")
            edge.catalogue_validated = True
            return
        if not task_keys:
            raise BackendContractError("nonzero edge catalogue cannot be empty")

        records = sorted(
            (self.tasks_by_key[key] for key in task_keys),
            key=lambda item: (item.byte_offset, item.chunk_index, item.task_stable_key),
        )
        expected_offset = 0
        seen_chunks: set[int] = set()
        for record in records:
            if record.edge_stable_key != edge.edge_stable_key:
                raise BackendContractError("task edge mismatch")
            if record.phase_stable_key != edge.phase_stable_key:
                raise BackendContractError("task phase mismatch")
            if record.src_rank != edge.src_rank or record.dst_rank != edge.dst_rank:
                raise BackendContractError("task source/destination mismatch")
            if record.chunk_index in seen_chunks:
                raise BackendContractError("duplicate chunk_index in edge catalogue")
            seen_chunks.add(record.chunk_index)
            if record.byte_offset != expected_offset:
                raise BackendContractError(
                    "task ranges must be contiguous, complete and non-overlapping"
                )
            expected_offset += record.payload_bytes
            capacity = self.memory_by_rank[record.dst_rank].capacity_bytes
            if capacity is not None and record.payload_bytes > capacity:
                raise CapacityConfigurationError(
                    "fixed staging capacity is smaller than a canonical task"
                )
        if expected_offset != edge.expected_bytes:
            raise BackendContractError(
                f"task ranges cover {expected_offset} bytes, expected {edge.expected_bytes}"
            )
        edge.task_ids = [record.task_id for record in records]
        edge.catalogue_validated = True

    # ------------------------------------------------------------------
    # Source readiness and job eligibility
    # ------------------------------------------------------------------
    def mark_source_payload_ready(
        self, *, phase_key: Any, src_rank: int, at_ns: int
    ) -> None:
        if not isinstance(src_rank, int) or isinstance(src_rank, bool):
            raise BackendContractError("source payload rank must be an int")
        if src_rank < 0 or src_rank >= self.world_size:
            raise BackendContractError("source payload rank is outside world_size")
        at_ns = require_time_ns(at_ns, field="source_payload_ready.at_ns")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        key = (phase_stable_key, src_rank)
        existing = self._source_payload_ready_at.get(key)
        if existing is not None:
            if existing != at_ns:
                raise DuplicateRegistrationError(
                    "source payload readiness changed after publication"
                )
            return
        self._source_payload_ready_at[key] = at_ns
        self.observer.emit(
            kind="SOURCE_PAYLOAD_READY",
            at_ns=at_ns,
            payload={"phase_key": phase_key, "src_rank": src_rank},
        )
        for edge in sorted(
            self.edges_by_key.values(), key=lambda item: item.edge_stable_key
        ):
            if edge.phase_stable_key == phase_stable_key and edge.src_rank == src_rank:
                self._refresh_edge_jobs(edge, now_ns=at_ns)
                self._refresh_local_assembly(edge, now_ns=at_ns)

    def _refresh_local_assembly(self, edge: EdgeState, *, now_ns: int) -> None:
        if (
            self.allow_legacy_local_network_tasks_for_unit_tests
            or edge.zero_edge
            or edge.src_rank != edge.dst_rank
        ):
            return
        if edge.data_complete_at_ns is not None or edge.edge_stable_key in self._local_assembly_scheduled:
            return
        payload_ready = self._source_payload_ready_at.get(
            (edge.phase_stable_key, edge.src_rank)
        )
        if payload_ready is None:
            return
        start_at = max(int(now_ns), int(edge.expectation_available_at_ns), int(payload_ready))
        finish_at = start_at + self.local_assembly_cost_ns
        self._local_assembly_scheduled.add(edge.edge_stable_key)
        self.observer.emit(
            kind="LOCAL_ASSEMBLY_STARTED",
            at_ns=start_at,
            payload={
                "edge_key": edge.edge_key,
                "phase_key": edge.phase_key,
                "src_rank": edge.src_rank,
                "dst_rank": edge.dst_rank,
                "bytes": edge.expected_bytes,
                "local_assembly_cost_ns": self.local_assembly_cost_ns,
            },
        )
        self.kernel.schedule_backend_event(
            time_ns=finish_at,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_LOCAL_ASSEMBLY_FINISH,
                time_ns=finish_at,
                semantic_parts=[edge.edge_stable_key],
            ),
            event_kind=_EVENT_LOCAL_ASSEMBLY_FINISH,
            payload={"edge_key": edge.edge_stable_key},
        )

    def on_local_assembly_finish(self, *, edge_key: str, at_ns: int) -> EdgeState:
        at_ns = require_time_ns(at_ns, field="local_assembly_finish.at_ns")
        try:
            edge = self.edges_by_key[str(edge_key)]
        except KeyError as exc:
            raise UnknownObjectError(f"unknown local edge {edge_key}") from exc
        if edge.src_rank != edge.dst_rank or edge.zero_edge:
            raise IllegalTransitionError("local assembly finish requires nonzero diagonal edge")
        if edge.data_complete_at_ns is not None:
            raise IllegalTransitionError("local edge assembled more than once")
        memory = self.memory_by_rank[edge.dst_rank]
        memory.final_assembly_bytes += edge.expected_bytes
        self._final_assembly_by_phase_dst[(edge.phase_stable_key, edge.dst_rank)] += edge.expected_bytes
        phase_metrics = self._phase_metrics(
            phase_stable_key=edge.phase_stable_key, dst_rank=edge.dst_rank
        )
        phase_metrics.current_final_assembly_bytes += edge.expected_bytes
        memory.update_peaks()
        phase_metrics.update_peaks()
        self._record_memory_state(
            phase_stable_key=edge.phase_stable_key, dst_rank=edge.dst_rank, at_ns=at_ns
        )
        edge.assembled_bytes = edge.expected_bytes
        edge.data_complete_at_ns = at_ns
        self.observer.emit(
            kind="LOCAL_ASSEMBLY_COMPLETE",
            at_ns=at_ns,
            payload={
                "edge_key": edge.edge_key,
                "phase_key": edge.phase_key,
                "src_rank": edge.src_rank,
                "dst_rank": edge.dst_rank,
                "bytes": edge.expected_bytes,
            },
        )
        self.observer.emit(
            kind="EDGE_FINAL_ASSEMBLY_COMPLETE",
            at_ns=at_ns,
            payload={
                "edge_key": edge.edge_key,
                "phase_key": edge.phase_key,
                "src_rank": edge.src_rank,
                "dst_rank": edge.dst_rank,
                "bytes": edge.expected_bytes,
                "local": True,
            },
        )
        return edge

    def _refresh_edge_jobs(self, edge: EdgeState, *, now_ns: int) -> None:
        if edge.zero_edge or not edge.catalogue_validated:
            return
        payload_ready = self._source_payload_ready_at.get(
            (edge.phase_stable_key, edge.src_rank)
        )
        for task_key in self._pending_task_keys_by_edge[edge.edge_stable_key]:
            task = self.tasks_by_key[task_key]
            requested_at = max(
                edge.expectation_available_at_ns, task.registered_at_ns
            )
            if task.requested_at_ns is None:
                task.requested_at_ns = requested_at
                self.observer.emit(
                    kind="RECEIVER_JOB_REQUESTED",
                    at_ns=requested_at,
                    payload={
                        "task_id": task.task_id,
                        "edge_key": task.edge_key,
                        "requested_at_ns": requested_at,
                    },
                )
            if payload_ready is None:
                continue
            eligible_at = max(requested_at, payload_ready)
            if task.eligible_at_ns is None:
                task.eligible_at_ns = eligible_at
                task.status = ReceiverJobStatus.ELIGIBLE
                self.posting_by_rank[task.dst_rank].eligible_task_ids.add(task.task_id)
                self.observer.emit(
                    kind="RECEIVER_JOB_ELIGIBLE",
                    at_ns=eligible_at,
                    payload={
                        "task_id": task.task_id,
                        "edge_key": task.edge_key,
                        "eligible_at_ns": eligible_at,
                    },
                )

    # ------------------------------------------------------------------
    # Posting service and Permit creation
    # ------------------------------------------------------------------
    def stabilize_posting(self, *, dst_rank: int, now_ns: int) -> bool:
        now_ns = require_time_ns(now_ns, field="stabilize_posting.now_ns")
        server = self.posting_by_rank[dst_rank]
        if server.active_task_id is not None or server.available_at_ns > now_ns:
            return False
        candidates = [
            self.task_record(task_id)
            for task_id in server.eligible_task_ids
            if self.task_record(task_id).status == ReceiverJobStatus.ELIGIBLE
            and self.task_record(task_id).eligible_at_ns is not None
            and self.task_record(task_id).eligible_at_ns <= now_ns
        ]
        if not candidates:
            return False
        head = min(candidates, key=lambda item: item.fifo_key())
        memory = self.memory_by_rank[dst_rank]
        free = memory.free_bytes
        if free is not None and free < head.payload_bytes:
            if head.buffer_stall_started_ns is None:
                head.buffer_stall_started_ns = now_ns
                self.observer.emit(
                    kind="RECEIVER_BUFFER_HOL_STALL_BEGIN",
                    at_ns=now_ns,
                    payload={
                        "task_id": head.task_id,
                        "dst_rank": dst_rank,
                        "required_bytes": head.payload_bytes,
                        "free_bytes": free,
                    },
                )
            return False

        if head.buffer_stall_started_ns is not None:
            stalled = now_ns - head.buffer_stall_started_ns
            head.buffer_stall_ns += stalled
            memory.receiver_buffer_stall_ns += stalled
            self._phase_metrics(
                phase_stable_key=head.phase_stable_key, dst_rank=dst_rank
            ).receiver_buffer_stall_ns += stalled
            self.observer.emit(
                kind="RECEIVER_BUFFER_HOL_STALL_END",
                at_ns=now_ns,
                payload={
                    "task_id": head.task_id,
                    "dst_rank": dst_rank,
                    "stall_ns": stalled,
                },
            )
            head.buffer_stall_started_ns = None

        assert head.eligible_at_ns is not None
        total_eligibility_wait = now_ns - head.eligible_at_ns
        posting_queue_wait = total_eligibility_wait - int(head.buffer_stall_ns)
        if total_eligibility_wait < 0 or posting_queue_wait < 0:
            raise IllegalTransitionError(
                "receiver posting queue wait is inconsistent with task eligibility/buffer stall"
            )
        memory.receiver_posting_queue_wait_ns += posting_queue_wait
        phase_metrics = self._phase_metrics(
            phase_stable_key=head.phase_stable_key, dst_rank=dst_rank
        )
        phase_metrics.receiver_posting_queue_wait_ns += posting_queue_wait
        memory.reserved_bytes += head.payload_bytes
        phase_metrics.current_staging_bytes += head.payload_bytes
        memory.update_peaks()
        phase_metrics.update_peaks()
        self._record_memory_state(
            phase_stable_key=head.phase_stable_key, dst_rank=dst_rank, at_ns=now_ns
        )
        head.reservation_id = f"receiver-reservation:{head.task_stable_key}"
        head.receiver_start_ns = now_ns
        head.status = ReceiverJobStatus.POSTING
        server.active_task_id = head.task_id
        server.eligible_task_ids.discard(head.task_id)
        duration = require_nonnegative_int(
            self.cost_model.receiver_service_cost_ns(head.payload_bytes),
            field="receiver_service_cost_ns",
        )
        posted_at = now_ns + duration
        head.receive_posted_at_ns = posted_at
        server.available_at_ns = posted_at
        self.observer.emit(
            kind="RECEIVER_POSTING_STARTED",
            at_ns=now_ns,
            payload={
                "task_id": head.task_id,
                "dst_rank": dst_rank,
                "reservation_id": head.reservation_id,
                "task_bytes": head.payload_bytes,
            },
        )
        self.kernel.schedule_backend_event(
            time_ns=posted_at,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_RECEIVER_POST_COMPLETE,
                time_ns=posted_at,
                semantic_parts=[head.task_stable_key],
            ),
            event_kind=_EVENT_RECEIVER_POST_COMPLETE,
            payload={"task_key": head.task_stable_key},
        )
        return True

    def on_receiver_post_complete(self, *, task_key: str, at_ns: int) -> Any:
        at_ns = require_time_ns(at_ns, field="receiver_post_complete.at_ns")
        task = self._task_by_key(task_key)
        if task.status != ReceiverJobStatus.POSTING:
            raise IllegalTransitionError("posting completion requires POSTING task")
        if task.receive_posted_at_ns != at_ns:
            raise IllegalTransitionError("posting completion timestamp mismatch")
        server = self.posting_by_rank[task.dst_rank]
        if self.adapter.stable_key(server.active_task_id) != self.adapter.stable_key(
            task.task_id
        ):
            raise IllegalTransitionError("posting server active task mismatch")
        edge = self.edges_by_key[task.edge_stable_key]
        if task.receiver_start_ns is None:
            raise IllegalTransitionError("posting completion is missing receiver start")
        post_wait = at_ns - task.receiver_start_ns
        if post_wait < 0:
            raise IllegalTransitionError("posting completion preceded receiver start")
        self.memory_by_rank[task.dst_rank].receiver_posting_service_ns += post_wait
        self._phase_metrics(
            phase_stable_key=task.phase_stable_key, dst_rank=task.dst_rank
        ).receiver_posting_service_ns += post_wait
        task.status = ReceiverJobStatus.POSTED
        server.active_task_id = None
        permit_id = f"receiver-permit:{task.task_stable_key}"
        task.permit_object = self.permit_factory.create_receive_permit(
            permit_id=permit_id,
            task_id=task.task_id,
            edge_key=task.edge_key,
            chunk_index=task.chunk_index,
            byte_offset=task.byte_offset,
            task_bytes=task.payload_bytes,
            credit_reservation_id=task.reservation_id or "",
            expectation_digest=edge.expectation_digest,
            descriptor_digest_or_none=edge.descriptor_digest_or_none,
            posted_at_ns=at_ns,
        )
        self.observer.emit(
            kind="RECEIVE_PERMIT_GRANTED",
            at_ns=at_ns,
            payload={
                "permit": task.permit_object,
                "task_id": task.task_id,
                "edge_key": task.edge_key,
            },
        )
        return task.permit_object

    # ------------------------------------------------------------------
    # Transfer completion and drain line
    # ------------------------------------------------------------------
    def on_transfer_completed(self, *, task_id: Any, at_ns: int) -> None:
        at_ns = require_time_ns(at_ns, field="transfer_completed.at_ns")
        task = self.task_record(task_id)
        if task.status != ReceiverJobStatus.POSTED:
            raise IllegalTransitionError(
                "network completion requires a posted task-level ReceivePermit"
            )
        if task.receive_posted_at_ns is None or at_ns < task.receive_posted_at_ns:
            raise IllegalTransitionError(
                "network completion cannot precede ReceivePermit posting"
            )
        memory = self.memory_by_rank[task.dst_rank]
        if memory.reserved_bytes < task.payload_bytes:
            raise IllegalTransitionError("reserved staging underflow")
        memory.reserved_bytes -= task.payload_bytes
        memory.used_bytes += task.payload_bytes
        memory.update_peaks()
        self._record_memory_state(
            phase_stable_key=task.phase_stable_key, dst_rank=task.dst_rank, at_ns=at_ns
        )
        task.transfer_complete_at_ns = at_ns
        task.status = ReceiverJobStatus.TRANSFER_COMPLETED
        drain = self.drain_by_rank[task.dst_rank]
        drain.waiting_task_ids.add(task.task_id)
        self.observer.emit(
            kind="TRANSFER_COMPLETED_BACKEND_APPLIED",
            at_ns=at_ns,
            payload={"task_id": task.task_id, "dst_rank": task.dst_rank},
        )

    def stabilize_drain(self, *, dst_rank: int, now_ns: int) -> bool:
        now_ns = require_time_ns(now_ns, field="stabilize_drain.now_ns")
        drain = self.drain_by_rank[dst_rank]
        if drain.active_task_id is not None or drain.available_at_ns > now_ns:
            return False
        candidates = [
            self.task_record(task_id)
            for task_id in drain.waiting_task_ids
            if self.task_record(task_id).status
            == ReceiverJobStatus.TRANSFER_COMPLETED
            and self.task_record(task_id).transfer_complete_at_ns is not None
            and self.task_record(task_id).transfer_complete_at_ns <= now_ns
        ]
        if not candidates:
            return False
        task = min(
            candidates,
            key=lambda item: (
                item.transfer_complete_at_ns,
                item.phase_stable_key,
                item.src_rank,
                item.chunk_index,
                item.task_stable_key,
            ),
        )
        task.drain_start_ns = now_ns
        transfer_complete_at = task.transfer_complete_at_ns
        if transfer_complete_at is None:
            raise IllegalTransitionError("drain task is missing transfer completion time")
        task.drain_queue_wait_ns = now_ns - transfer_complete_at
        self.memory_by_rank[dst_rank].receiver_drain_queue_wait_ns += task.drain_queue_wait_ns
        self._phase_metrics(
            phase_stable_key=task.phase_stable_key, dst_rank=dst_rank
        ).receiver_drain_queue_wait_ns += task.drain_queue_wait_ns
        duration = require_nonnegative_int(
            self.cost_model.receiver_drain_cost_ns(task.payload_bytes),
            field="receiver_drain_cost_ns",
        )
        finish = now_ns + duration
        task.drain_finish_ns = finish
        task.status = ReceiverJobStatus.DRAINING
        drain.active_task_id = task.task_id
        drain.waiting_task_ids.discard(task.task_id)
        drain.available_at_ns = finish
        self.observer.emit(
            kind="RECEIVER_DRAIN_STARTED",
            at_ns=now_ns,
            payload={
                "task_id": task.task_id,
                "dst_rank": dst_rank,
                "drain_queue_wait_ns": task.drain_queue_wait_ns,
            },
        )
        self.kernel.schedule_backend_event(
            time_ns=finish,
            phase_priority=BACKEND_PHASE_PRIORITY,
            stable_event_id=stable_semantic_event_id(
                event_kind=_EVENT_RECEIVER_DRAIN_FINISH,
                time_ns=finish,
                semantic_parts=[task.task_stable_key],
            ),
            event_kind=_EVENT_RECEIVER_DRAIN_FINISH,
            payload={"task_key": task.task_stable_key},
        )
        return True

    def on_receiver_drain_finish(
        self, *, task_key: str, at_ns: int
    ) -> EdgeState | None:
        at_ns = require_time_ns(at_ns, field="receiver_drain_finish.at_ns")
        task = self._task_by_key(task_key)
        if task.status != ReceiverJobStatus.DRAINING:
            raise IllegalTransitionError("drain finish requires DRAINING task")
        if task.drain_finish_ns != at_ns:
            raise IllegalTransitionError("drain finish timestamp mismatch")
        drain = self.drain_by_rank[task.dst_rank]
        if self.adapter.stable_key(drain.active_task_id) != self.adapter.stable_key(
            task.task_id
        ):
            raise IllegalTransitionError("drain line active task mismatch")
        if task.drain_start_ns is None:
            raise IllegalTransitionError("drain finish is missing drain start")
        drain_service_ns = at_ns - int(task.drain_start_ns)
        if drain_service_ns < 0:
            raise IllegalTransitionError("drain finish preceded drain start")
        memory = self.memory_by_rank[task.dst_rank]
        memory.receiver_drain_service_ns += drain_service_ns
        self._phase_metrics(
            phase_stable_key=task.phase_stable_key, dst_rank=task.dst_rank
        ).receiver_drain_service_ns += drain_service_ns
        if memory.used_bytes < task.payload_bytes:
            raise IllegalTransitionError("used staging underflow")
        memory.used_bytes -= task.payload_bytes
        memory.final_assembly_bytes += task.payload_bytes
        phase_dst = (task.phase_stable_key, task.dst_rank)
        self._final_assembly_by_phase_dst[phase_dst] += task.payload_bytes
        phase_metrics = self._phase_metrics(
            phase_stable_key=task.phase_stable_key, dst_rank=task.dst_rank
        )
        if phase_metrics.current_staging_bytes < task.payload_bytes:
            raise IllegalTransitionError("phase staging accounting underflow")
        phase_metrics.current_staging_bytes -= task.payload_bytes
        phase_metrics.current_final_assembly_bytes += task.payload_bytes
        memory.update_peaks()
        phase_metrics.update_peaks()
        self._record_memory_state(
            phase_stable_key=task.phase_stable_key, dst_rank=task.dst_rank, at_ns=at_ns
        )
        task.status = ReceiverJobStatus.ASSEMBLED
        drain.active_task_id = None

        edge = self.edges_by_key[task.edge_stable_key]
        edge.assembled_bytes += task.payload_bytes
        completed_edge: EdgeState | None = None
        if edge.assembled_bytes > edge.expected_bytes:
            raise IllegalTransitionError("assembled bytes exceeded expectation")
        if edge.assembled_bytes == edge.expected_bytes:
            edge.data_complete_at_ns = at_ns
            completed_edge = edge
            self.observer.emit(
                kind="EDGE_FINAL_ASSEMBLY_COMPLETE",
                at_ns=at_ns,
                payload={
                    "edge_key": edge.edge_key,
                    "phase_key": edge.phase_key,
                    "src_rank": edge.src_rank,
                    "dst_rank": edge.dst_rank,
                    "bytes": edge.expected_bytes,
                },
            )
        self.observer.emit(
            kind="RECEIVER_DRAIN_FINISHED",
            at_ns=at_ns,
            payload={"task_id": task.task_id, "dst_rank": task.dst_rank},
        )

        return completed_edge

    def stabilize(self, *, now_ns: int) -> bool:
        """Run one deterministic receiver-service stabilization pass.

        External phase-1/2/3 updates must be collected before this phase-4 pass.
        At most one posting job and one drain job can start per destination.
        """
        now_ns = require_time_ns(now_ns, field="receiver_stabilize.now_ns")
        progressed = False
        for dst_rank in range(self.world_size):
            progressed = self.stabilize_drain(
                dst_rank=dst_rank, now_ns=now_ns
            ) or progressed
            progressed = self.stabilize_posting(
                dst_rank=dst_rank, now_ns=now_ns
            ) or progressed
        return progressed

    # ------------------------------------------------------------------
    # Completion predicates and final assembly lifecycle
    # ------------------------------------------------------------------
    def expected_edges_for_destination(
        self, *, phase_key: Any, dst_rank: int
    ) -> list[EdgeState]:
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        return list(self._edges_by_phase_dst.get((phase_stable_key, dst_rank), ()))

    def all_nonzero_inbound_assembled(
        self, *, phase_key: Any, dst_rank: int
    ) -> tuple[bool, int | None]:
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        cache_key = (phase_stable_key, dst_rank)
        cached = self._assembled_completion_cache.get(cache_key)
        if cached is not None:
            return cached
        edges = self.expected_edges_for_destination(
            phase_key=phase_key, dst_rank=dst_rank
        )
        phase_kind = self.phase_semantics.phase_kind(phase_key)
        if phase_kind == "COMBINE":
            if (phase_stable_key, dst_rank) not in self._sealed_combine_destinations:
                return False, None
        else:
            # A Dispatch destination cannot prove that all nonzero inbound data is
            # assembled until every source row has created its edge expectation.
            # Individual transfers may still complete before descriptor closure;
            # their exact completion times remain recorded and are used once the
            # complete world-size expectation set is available.
            if len(edges) != self.world_size:
                return False, None
        if not edges:
            return False, None
        nonzero = [edge for edge in edges if not edge.zero_edge]
        if any(not edge.catalogue_validated for edge in nonzero):
            return False, None
        if any(edge.data_complete_at_ns is None for edge in nonzero):
            return False, None
        if nonzero:
            result = (True, max(edge.data_complete_at_ns or 0 for edge in nonzero))
        else:
            # All-zero destination: expectation/descriptor closure is the data proof.
            result = (True, max(edge.expectation_available_at_ns for edge in edges))
        self._assembled_completion_cache[cache_key] = result
        return result

    def release_final_assembly(
        self, *, phase_key: Any, dst_rank: int, at_ns: int
    ) -> int:
        at_ns = require_time_ns(at_ns, field="release_final_assembly.at_ns")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        key = (phase_stable_key, dst_rank)
        if key in self._released_final_assembly:
            raise IllegalTransitionError("final assembly already released")
        complete, _ = self.all_nonzero_inbound_assembled(
            phase_key=phase_key, dst_rank=dst_rank
        )
        if not complete:
            raise IllegalTransitionError(
                "cannot release final assembly before inbound completion proof"
            )
        bytes_to_release = self._final_assembly_by_phase_dst.get(key, 0)
        memory = self.memory_by_rank[dst_rank]
        if memory.final_assembly_bytes < bytes_to_release:
            raise IllegalTransitionError("final assembly accounting underflow")
        memory.final_assembly_bytes -= bytes_to_release
        phase_metrics = self._phase_metrics(
            phase_stable_key=phase_stable_key, dst_rank=dst_rank
        )
        if phase_metrics.current_final_assembly_bytes < bytes_to_release:
            raise IllegalTransitionError("phase final assembly accounting underflow")
        phase_metrics.current_final_assembly_bytes -= bytes_to_release
        memory.update_peaks()
        phase_metrics.update_peaks()
        self._record_memory_state(
            phase_stable_key=phase_stable_key, dst_rank=dst_rank, at_ns=at_ns
        )
        self._released_final_assembly.add(key)
        self.observer.emit(
            kind="FINAL_ASSEMBLY_RELEASED",
            at_ns=at_ns,
            payload={
                "phase_key": phase_key,
                "dst_rank": dst_rank,
                "released_bytes": bytes_to_release,
            },
        )
        return bytes_to_release


    def phase_terminal_snapshot(self, *, phase_key: Any) -> dict[str, Any]:
        """Return an auditable receiver terminal snapshot for one phase.

        A closed phase has the finalized expectation set, exact byte
        reconciliation, no outstanding task/Permit or receiver job, no active
        posting/drain service, and no phase-owned staging/final assembly bytes.
        """

        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        edges = sorted(
            [
                edge
                for edge in self.edges_by_key.values()
                if edge.phase_stable_key == phase_stable_key
            ],
            key=lambda edge: (edge.dst_rank, edge.src_rank, edge.edge_stable_key),
        )
        tasks = sorted(
            [
                task
                for task in self.tasks_by_key.values()
                if task.phase_stable_key == phase_stable_key
            ],
            key=lambda task: task.task_stable_key,
        )
        incomplete_edges = tuple(
            edge.edge_stable_key
            for edge in edges
            if not edge.zero_edge and edge.data_complete_at_ns is None
        )
        byte_mismatch_edges = tuple(
            edge.edge_stable_key
            for edge in edges
            if int(edge.assembled_bytes) != int(edge.expected_bytes)
        )
        outstanding_tasks = tuple(
            task.task_stable_key
            for task in tasks
            if task.status != ReceiverJobStatus.ASSEMBLED
        )
        unconsumed_permits = tuple(
            task.task_stable_key
            for task in tasks
            if task.permit_object is not None
            and task.status != ReceiverJobStatus.ASSEMBLED
        )
        final_by_rank = {
            rank: int(
                self._final_assembly_by_phase_dst.get(
                    (phase_stable_key, rank), 0
                )
            )
            if (phase_stable_key, rank) not in self._released_final_assembly
            else 0
            for rank in range(self.world_size)
        }
        staging_by_rank = {
            rank: int(
                self._phase_metrics(
                    phase_stable_key=phase_stable_key, dst_rank=rank
                ).current_staging_bytes
            )
            for rank in range(self.world_size)
        }
        active_posting = tuple(
            rank
            for rank, server in sorted(self.posting_by_rank.items())
            if server.active_task_id is not None
            and self.task_record(server.active_task_id).phase_stable_key
            == phase_stable_key
        )
        active_drain = tuple(
            rank
            for rank, line in sorted(self.drain_by_rank.items())
            if line.active_task_id is not None
            and self.task_record(line.active_task_id).phase_stable_key
            == phase_stable_key
        )
        total_expected_bytes = sum(int(edge.expected_bytes) for edge in edges)
        total_assembled_bytes = sum(int(edge.assembled_bytes) for edge in edges)
        expectation_closure_finalized = (
            phase_stable_key in self._finalized_expectation_closure_by_phase
        )
        closed = bool(edges) and expectation_closure_finalized and not any(
            (
                incomplete_edges,
                byte_mismatch_edges,
                outstanding_tasks,
                unconsumed_permits,
                active_posting,
                active_drain,
                tuple(
                    rank for rank, value in staging_by_rank.items() if value != 0
                ),
                tuple(rank for rank, value in final_by_rank.items() if value != 0),
            )
        )
        return {
            "phase_key": phase_key,
            "expectation_count": len(edges),
            "task_count": len(tasks),
            "total_expected_bytes": total_expected_bytes,
            "total_assembled_bytes": total_assembled_bytes,
            "bytes_reconciled": (
                total_expected_bytes == total_assembled_bytes
                and not byte_mismatch_edges
            ),
            "expectation_closure_finalized": expectation_closure_finalized,
            "expectation_closure_digest": (
                self._finalized_expectation_closure_by_phase.get(
                    phase_stable_key
                )
            ),
            "incomplete_edges": incomplete_edges,
            "byte_mismatch_edges": byte_mismatch_edges,
            "outstanding_tasks": outstanding_tasks,
            "outstanding_receiver_jobs": outstanding_tasks,
            "unconsumed_permits": unconsumed_permits,
            "active_posting_ranks": active_posting,
            "active_drain_ranks": active_drain,
            "staging_bytes_per_rank": staging_by_rank,
            "final_assembly_bytes_per_rank": final_by_rank,
            "closed": closed,
        }

    def metrics_reconciliation_snapshot(self) -> dict[str, Any]:
        """Reconcile exported aggregate and per-phase receiver accounting."""

        phase_tokens = sorted(
            {
                phase_token
                for phase_token, _ in self._phase_metrics_by_phase_dst
            }
        )
        rows: dict[int, dict[str, int | bool]] = {}
        all_reconciled = True
        for rank in range(self.world_size):
            memory = self.memory_by_rank[rank]
            phase_metrics = [
                self._phase_metrics_by_phase_dst[(phase_token, rank)]
                for phase_token in phase_tokens
                if (phase_token, rank) in self._phase_metrics_by_phase_dst
            ]
            phase_staging = sum(item.current_staging_bytes for item in phase_metrics)
            phase_final = sum(
                item.current_final_assembly_bytes for item in phase_metrics
            )
            phase_post = sum(item.receiver_posting_service_ns for item in phase_metrics)
            phase_credit = sum(
                item.receiver_posting_queue_wait_ns for item in phase_metrics
            )
            phase_stall = sum(
                item.receiver_buffer_stall_ns for item in phase_metrics
            )
            phase_drain_queue = sum(
                item.receiver_drain_queue_wait_ns for item in phase_metrics
            )
            phase_drain_service = sum(
                item.receiver_drain_service_ns for item in phase_metrics
            )
            aggregate_peak_bounds_valid = bool(
                max(
                    memory.peak_staging_bytes,
                    memory.peak_final_assembly_bytes,
                )
                <= memory.peak_total_receiver_bytes
                <= memory.peak_staging_bytes
                + memory.peak_final_assembly_bytes
            )
            phase_peak_bounds_valid = all(
                max(
                    item.peak_staging_bytes,
                    item.peak_final_assembly_bytes,
                )
                <= item.peak_total_receiver_bytes
                <= item.peak_staging_bytes
                + item.peak_final_assembly_bytes
                for item in phase_metrics
            )
            reconciled = bool(
                phase_staging == memory.staging_bytes
                and phase_final == memory.final_assembly_bytes
                and phase_post == memory.receiver_posting_service_ns
                and phase_credit == memory.receiver_posting_queue_wait_ns
                and phase_stall == memory.receiver_buffer_stall_ns
                and phase_drain_queue == memory.receiver_drain_queue_wait_ns
                and phase_drain_service == memory.receiver_drain_service_ns
                and aggregate_peak_bounds_valid
                and phase_peak_bounds_valid
            )
            all_reconciled = all_reconciled and reconciled
            rows[rank] = {
                "aggregate_staging_bytes": int(memory.staging_bytes),
                "phase_staging_bytes": int(phase_staging),
                "aggregate_final_assembly_bytes": int(
                    memory.final_assembly_bytes
                ),
                "phase_final_assembly_bytes": int(phase_final),
                "aggregate_receiver_posting_service_ns": int(
                    memory.receiver_posting_service_ns
                ),
                "phase_receiver_posting_service_ns": int(phase_post),
                "aggregate_receiver_posting_queue_wait_ns": int(
                    memory.receiver_posting_queue_wait_ns
                ),
                "phase_receiver_posting_queue_wait_ns": int(phase_credit),
                "aggregate_receiver_buffer_stall_ns": int(
                    memory.receiver_buffer_stall_ns
                ),
                "phase_receiver_buffer_stall_ns": int(phase_stall),
                "aggregate_receiver_drain_queue_wait_ns": int(
                    memory.receiver_drain_queue_wait_ns
                ),
                "phase_receiver_drain_queue_wait_ns": int(phase_drain_queue),
                "aggregate_receiver_drain_service_ns": int(
                    memory.receiver_drain_service_ns
                ),
                "phase_receiver_drain_service_ns": int(phase_drain_service),
                "aggregate_peak_bounds_valid": aggregate_peak_bounds_valid,
                "phase_peak_bounds_valid": phase_peak_bounds_valid,
                "reconciled": reconciled,
            }
        return {"reconciled": all_reconciled, "per_rank": rows}

    def _record_memory_state(
        self, *, phase_stable_key: str, dst_rank: int, at_ns: int
    ) -> None:
        memory = self.memory_by_rank[int(dst_rank)]
        self._memory_journal.append((
            int(at_ns), str(phase_stable_key), int(dst_rank),
            int(memory.staging_bytes), int(memory.final_assembly_bytes),
            int(memory.staging_bytes + memory.final_assembly_bytes),
        ))

    def window_memory_peaks(
        self, *, phase_keys: Iterable[Any]
    ) -> dict[int, dict[str, int]]:
        phase_ids = {self.phase_semantics.phase_sort_key(key) for key in phase_keys}
        result = {
            rank: {"staging": 0, "final_assembly": 0, "total": 0}
            for rank in range(self.world_size)
        }
        for _at_ns, phase_id, rank, staging, final_assembly, total in self._memory_journal:
            if phase_id not in phase_ids:
                continue
            row = result[rank]
            row["staging"] = max(row["staging"], staging)
            row["final_assembly"] = max(row["final_assembly"], final_assembly)
            row["total"] = max(row["total"], total)
        return result

    def _phase_metrics(
        self, *, phase_stable_key: str, dst_rank: int
    ) -> PhaseDestinationMetrics:
        key = (str(phase_stable_key), int(dst_rank))
        metrics = self._phase_metrics_by_phase_dst.get(key)
        if metrics is None:
            metrics = PhaseDestinationMetrics()
            self._phase_metrics_by_phase_dst[key] = metrics
        return metrics

    def expectation_closure_at_ns(
        self, *, phase_key: Any, dst_rank: int
    ) -> int | None:
        """Return complete expectation availability for one destination."""

        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        edges = self.expected_edges_for_destination(
            phase_key=phase_key, dst_rank=dst_rank
        )
        if len(edges) != self.world_size:
            return None
        if self.phase_semantics.phase_kind(phase_key) == "COMBINE":
            return self._sealed_combine_destinations.get(
                (phase_stable_key, dst_rank)
            )
        return max(edge.expectation_available_at_ns for edge in edges)

    def phase_rank_receiver_metrics(
        self, *, phase_key: Any, dst_rank: int
    ) -> dict[str, int]:
        if not isinstance(dst_rank, int) or isinstance(dst_rank, bool):
            raise BackendContractError("destination rank must be an int")
        if dst_rank < 0 or dst_rank >= self.world_size:
            raise BackendContractError("destination rank is outside world_size")
        phase_stable_key = self.phase_semantics.phase_sort_key(phase_key)
        metrics = self._phase_metrics(
            phase_stable_key=phase_stable_key, dst_rank=dst_rank
        )
        return {
            "receiver_posting_service_ns": metrics.receiver_posting_service_ns,
            "receiver_posting_queue_wait_ns": metrics.receiver_posting_queue_wait_ns,
            "receiver_buffer_stall_ns": metrics.receiver_buffer_stall_ns,
            "receiver_drain_queue_wait_ns": metrics.receiver_drain_queue_wait_ns,
            "receiver_drain_service_ns": metrics.receiver_drain_service_ns,
            "peak_staging_bytes": metrics.peak_staging_bytes,
            "peak_final_assembly_bytes": metrics.peak_final_assembly_bytes,
            "peak_total_receiver_bytes": metrics.peak_total_receiver_bytes,
            "current_staging_bytes": metrics.current_staging_bytes,
            "current_final_assembly_bytes": (
                metrics.current_final_assembly_bytes
            ),
        }

    def metrics_snapshot(self) -> ReceiverMetricsSnapshot:
        return snapshot_memory(self.memory_by_rank)

    def current_memory(self, dst_rank: int) -> dict[str, int | None]:
        memory = self.memory_by_rank[dst_rank]
        return {
            "capacity_bytes": memory.capacity_bytes,
            "reserved_bytes": memory.reserved_bytes,
            "used_bytes": memory.used_bytes,
            "staging_bytes": memory.staging_bytes,
            "final_assembly_bytes": memory.final_assembly_bytes,
            "total_receiver_bytes": memory.staging_bytes
            + memory.final_assembly_bytes,
        }

    def task_record(self, task_id: Any) -> TaskStateRecord:
        external_key = self.adapter.stable_key(task_id)
        task_key = self._task_key_by_external_id.get(external_key)
        if task_key is None:
            raise UnknownObjectError(f"unknown task id {external_key}")
        return self.tasks_by_key[task_key]

    def receive_permit(self, task_id: Any) -> Any | None:
        """Public read-only Permit lookup used by the formal transport adapter."""
        return self.task_record(task_id).permit_object

    def _task_by_key(self, task_key: str) -> TaskStateRecord:
        try:
            return self.tasks_by_key[task_key]
        except KeyError as exc:
            raise UnknownObjectError(f"unknown internal task key {task_key}") from exc

    @property
    def event_kinds(self) -> frozenset[str]:
        return frozenset(
            {_EVENT_RECEIVER_POST_COMPLETE, _EVENT_RECEIVER_DRAIN_FINISH, _EVENT_LOCAL_ASSEMBLY_FINISH}
        )
