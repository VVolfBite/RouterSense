from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from rs_sim.contracts.factories import make_transfer_batch
from rs_sim.contracts.schema import (
    CommitReceipt,
    LinkClass,
    SubmitOutcome,
    TaskResourceFootprint,
    TransferBatch,
    TransportSnapshot,
)
from rs_sim.transport.api.ports import TaskLookupPort, TaskResourceResolverPort

from rs_sim.scheduler.execution.authority import PhaseAuthorityManager
from rs_sim.scheduler.planning.catalogue import TaskCatalogue
from rs_sim.scheduler.errors import CompilationError
from rs_sim.scheduler.planning.schema_api import CanonicalTaskView
from rs_sim.scheduler.stable import stable_id
from rs_sim.scheduler.execution.state import READY_UNCOMMITTED, TaskRuntimeIndex

# Public transport receives the immutable shared-schema object directly.  The legacy name remains
# as a source-compatible alias for scheduler imports, not as a second schema.
CompiledBatch = TransferBatch


@dataclass(frozen=True)
class CompileAttempt:
    code: str
    batch: TransferBatch | None
    ready_task_count: int


@dataclass(frozen=True)
class ValidationResult:
    code: str
    reason: str = ""


@dataclass(frozen=True)
class StabilizationResult:
    accepted_batch_digests: tuple[str, ...]
    accepted_task_ids: tuple[str, ...]
    terminal_code: str
    stale_retries: int


class BatchResourceAdapter(Protocol):
    def batch_limit(self, snapshot: Any) -> int: ...

    def can_add(
        self,
        snapshot: Any,
        selected: Sequence[CanonicalTaskView],
        candidate: CanonicalTaskView,
        *,
        now_ns: int,
    ) -> bool: ...

    def validate_batch(
        self,
        snapshot: Any,
        tasks: Sequence[CanonicalTaskView],
        *,
        now_ns: int,
    ) -> bool: ...

    def batch_link_class(
        self, snapshot: Any, tasks: Sequence[CanonicalTaskView]
    ) -> LinkClass: ...

    def topology_digest(self, snapshot: Any) -> str: ...


class EndpointConflictResourceAdapter:
    """Contract-fixture adapter; formal runs must use FormalTransportResourceAdapter."""

    def batch_limit(self, snapshot: Any) -> int:
        return int(getattr(snapshot, "max_batch_tasks", 1))

    @staticmethod
    def _busy(snapshot: Any, name: str) -> set[int]:
        return {int(item) for item in getattr(snapshot, name, ())}

    def can_add(
        self,
        snapshot: Any,
        selected: Sequence[CanonicalTaskView],
        candidate: CanonicalTaskView,
        *,
        now_ns: int,
    ) -> bool:
        del now_ns
        if candidate.src_rank in self._busy(snapshot, "busy_src_ranks"):
            return False
        if candidate.dst_rank in self._busy(snapshot, "busy_dst_ranks"):
            return False
        if any(item.src_rank == candidate.src_rank for item in selected):
            return False
        if any(item.dst_rank == candidate.dst_rank for item in selected):
            return False
        return True

    def validate_batch(
        self,
        snapshot: Any,
        tasks: Sequence[CanonicalTaskView],
        *,
        now_ns: int,
    ) -> bool:
        selected: list[CanonicalTaskView] = []
        if len(tasks) > self.batch_limit(snapshot):
            return False
        for task in tasks:
            if not self.can_add(snapshot, selected, task, now_ns=now_ns):
                return False
            selected.append(task)
        return bool(selected)

    def batch_link_class(
        self, snapshot: Any, tasks: Sequence[CanonicalTaskView]
    ) -> LinkClass:
        del tasks
        available = dict(getattr(snapshot, "available_lane_ids_by_link_class", ()))
        if available.get(LinkClass.INTER_NODE):
            return LinkClass.INTER_NODE
        if available.get(LinkClass.INTRA_NODE):
            return LinkClass.INTRA_NODE
        # Old fixture snapshots have no lane map.  This is explicitly non-formal.
        return LinkClass.INTER_NODE

    def topology_digest(self, snapshot: Any) -> str:
        return str(getattr(snapshot, "topology_digest", "CONTRACT_STUB_TOPOLOGY"))


class FormalTransportResourceAdapter:
    """Formal scheduler legality adapter over the shared transport-resource resolver.

    The same immutable ``TaskResourceResolverPort`` instance (or an equivalent
    object carrying the same frozen topology) is injected into transport and scheduler.  scheduler
    never reconstructs topology locally and validates both topology and
    hardware-profile digests before a batch can reach ``prepare_commit``.
    """

    def __init__(
        self,
        *,
        task_lookup: TaskLookupPort,
        resource_resolver: TaskResourceResolverPort,
        expected_hardware_profile_digest: str,
    ) -> None:
        topology = resource_resolver.topology
        if not str(topology.topology_digest):
            raise ValueError("resource resolver topology digest must be non-empty")
        if not str(expected_hardware_profile_digest):
            raise ValueError("expected_hardware_profile_digest must be non-empty")
        self.task_lookup = task_lookup
        self.resource_resolver = resource_resolver
        self.expected_topology_digest = str(topology.topology_digest)
        self.expected_hardware_profile_digest = str(expected_hardware_profile_digest)
        self._declared_lanes = {
            link_class: frozenset(lane_ids)
            for link_class, lane_ids in topology.lane_ids_by_link_class
        }
        self._declared_nics = frozenset(
            (*topology.tx_nic_id_by_rank, *topology.rx_nic_id_by_rank)
        )
        self._last_validated_snapshot: TransportSnapshot | None = None
        self._last_available_by_class: dict[LinkClass, tuple[str, ...]] = {}
        self._last_busy_src_ranks: frozenset[int] = frozenset()
        self._last_busy_dst_ranks: frozenset[int] = frozenset()
        self._last_busy_nic_ids: frozenset[str] = frozenset()
        self._footprint_cache: dict[str, TaskResourceFootprint] = {}

    def _require_snapshot(self, snapshot: Any) -> TransportSnapshot:
        if not isinstance(snapshot, TransportSnapshot):
            raise TypeError("FormalTransportResourceAdapter requires TransportSnapshot")
        if snapshot is self._last_validated_snapshot:
            return snapshot
        if snapshot.topology_digest != self.expected_topology_digest:
            raise CompilationError("FATAL_TOPOLOGY_CONTRACT_MISMATCH")
        if snapshot.hardware_profile_digest != self.expected_hardware_profile_digest:
            raise CompilationError("FATAL_HARDWARE_PROFILE_CONTRACT_MISMATCH")

        available = dict(snapshot.available_lane_ids_by_link_class)
        if set(available) - set(self._declared_lanes):
            raise CompilationError("FATAL_TOPOLOGY_CONTRACT_MISMATCH")
        for link_class, lane_ids in available.items():
            if not set(lane_ids).issubset(self._declared_lanes.get(link_class, frozenset())):
                raise CompilationError("FATAL_TOPOLOGY_CONTRACT_MISMATCH")
        declared_all_lanes = set().union(*self._declared_lanes.values()) if self._declared_lanes else set()
        if not set(snapshot.busy_lane_ids).issubset(declared_all_lanes):
            raise CompilationError("FATAL_TOPOLOGY_CONTRACT_MISMATCH")
        if not set(snapshot.busy_nic_ids).issubset(self._declared_nics):
            raise CompilationError("FATAL_TOPOLOGY_CONTRACT_MISMATCH")
        busy_lanes = frozenset(snapshot.busy_lane_ids)
        self._last_available_by_class = {
            link_class: tuple(lane for lane in lane_ids if lane not in busy_lanes)
            for link_class, lane_ids in snapshot.available_lane_ids_by_link_class
        }
        self._last_busy_src_ranks = frozenset(int(value) for value in snapshot.busy_src_ranks)
        self._last_busy_dst_ranks = frozenset(int(value) for value in snapshot.busy_dst_ranks)
        self._last_busy_nic_ids = frozenset(str(value) for value in snapshot.busy_nic_ids)
        self._last_validated_snapshot = snapshot
        return snapshot

    def batch_limit(self, snapshot: Any) -> int:
        return int(self._require_snapshot(snapshot).max_batch_tasks)

    def _available(self, snapshot: TransportSnapshot) -> dict[LinkClass, tuple[str, ...]]:
        self._require_snapshot(snapshot)
        return self._last_available_by_class

    def _footprints(
        self, tasks: Sequence[CanonicalTaskView]
    ) -> tuple[TaskResourceFootprint, ...]:
        footprints: list[TaskResourceFootprint] = []
        for view in tasks:
            cached = self._footprint_cache.get(view.task_id)
            if cached is not None:
                footprints.append(cached)
                continue
            task = self.task_lookup.task(view.task_id)
            if (
                task.task_id != view.task_id
                or task.phase_key != view.phase_key
                or task.src_rank != view.src_rank
                or task.dst_rank != view.dst_rank
                or task.chunk_index != view.chunk_index
                or task.byte_offset != view.byte_offset
                or task.payload_bytes != view.payload_bytes
                or task.expectation_digest != view.expectation_digest
            ):
                raise CompilationError("FATAL_TASK_LOOKUP_CONTRACT_MISMATCH")
            footprint = self.resource_resolver.footprint(task)
            if footprint.task_id != task.task_id:
                raise CompilationError("FATAL_TOPOLOGY_CONTRACT_MISMATCH")
            if footprint.topology_digest != self.expected_topology_digest:
                raise CompilationError("FATAL_TOPOLOGY_CONTRACT_MISMATCH")
            self._footprint_cache[view.task_id] = footprint
            footprints.append(footprint)
        return tuple(footprints)

    @staticmethod
    def _assignable_lanes(
        footprints: Sequence[TaskResourceFootprint],
        available: tuple[str, ...],
    ) -> bool:
        """Exact deterministic bipartite matching for task-to-lane legality."""

        available_set = set(available)
        choices_by_task = {
            item.task_id: tuple(sorted(set(item.eligible_lane_ids) & available_set))
            for item in footprints
        }
        if any(not choices for choices in choices_by_task.values()):
            return False

        lane_owner: dict[str, str] = {}

        def augment(task_id: str, visited: set[str]) -> bool:
            for lane_id in choices_by_task[task_id]:
                if lane_id in visited:
                    continue
                visited.add(lane_id)
                owner = lane_owner.get(lane_id)
                if owner is None or augment(owner, visited):
                    lane_owner[lane_id] = task_id
                    return True
            return False

        ordered = sorted(
            choices_by_task,
            key=lambda task_id: (len(choices_by_task[task_id]), task_id),
        )
        return all(augment(task_id, set()) for task_id in ordered)

    def can_add(
        self,
        snapshot: Any,
        selected: Sequence[CanonicalTaskView],
        candidate: CanonicalTaskView,
        *,
        now_ns: int,
    ) -> bool:
        del now_ns
        formal = self._require_snapshot(snapshot)
        if int(candidate.src_rank) == int(candidate.dst_rank):
            return False
        if int(candidate.src_rank) in self._last_busy_src_ranks:
            return False
        if int(candidate.dst_rank) in self._last_busy_dst_ranks:
            return False
        if any(int(item.src_rank) == int(candidate.src_rank) for item in selected):
            return False
        if any(int(item.dst_rank) == int(candidate.dst_rank) for item in selected):
            return False

        tasks = tuple((*selected, candidate))
        footprints = self._footprints(tasks)
        link_classes = {item.link_class for item in footprints}
        if len(link_classes) != 1 or LinkClass.LOCAL_ASSEMBLY in link_classes:
            return False
        busy_nics = self._last_busy_nic_ids
        all_nics: set[str] = set()
        for footprint in footprints:
            current = {footprint.tx_nic_id, footprint.rx_nic_id}
            if current & busy_nics or current & all_nics:
                return False
            all_nics.update(current)
        link_class = next(iter(link_classes))
        available = self._available(formal).get(link_class, ())
        return self._assignable_lanes(footprints, available)

    def validate_batch(
        self,
        snapshot: Any,
        tasks: Sequence[CanonicalTaskView],
        *,
        now_ns: int,
    ) -> bool:
        formal = self._require_snapshot(snapshot)
        if len(tasks) > int(formal.max_batch_tasks):
            return False
        selected: list[CanonicalTaskView] = []
        for task in tasks:
            if not self.can_add(formal, selected, task, now_ns=now_ns):
                return False
            selected.append(task)
        return bool(selected)

    def batch_link_class(
        self, snapshot: Any, tasks: Sequence[CanonicalTaskView]
    ) -> LinkClass:
        self._require_snapshot(snapshot)
        footprints = self._footprints(tasks)
        classes = {item.link_class for item in footprints}
        if len(classes) != 1 or LinkClass.LOCAL_ASSEMBLY in classes:
            raise CompilationError("batch contains mixed or local link classes")
        return next(iter(classes))

    def topology_digest(self, snapshot: Any) -> str:
        return self._require_snapshot(snapshot).topology_digest


class BatchCompiler:
    def __init__(
        self,
        *,
        catalogue: TaskCatalogue,
        runtime: TaskRuntimeIndex,
        authority: PhaseAuthorityManager,
        resources: BatchResourceAdapter | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.runtime = runtime
        self.authority = authority
        self.resources = resources or EndpointConflictResourceAdapter()

    def compile_next(
        self,
        *,
        phase_key: Any,
        snapshot: Any,
        now_ns: int,
        allowed_task_ids: Sequence[str] | None = None,
    ) -> CompileAttempt:
        active = self.authority.active_plan(phase_key)
        if active is None:
            return CompileAttempt(code="NO_ACTIVE_PLAN", batch=None, ready_task_count=0)
        plan_view = self.authority.adapter.plan_view(active)
        if plan_view.status != "ACTIVE":
            return CompileAttempt(code="NO_ACTIVE_PLAN", batch=None, ready_task_count=0)

        allowed = None if allowed_task_ids is None else tuple(str(item) for item in allowed_task_ids)
        if allowed is not None:
            if len(set(allowed)) != len(allowed):
                raise CompilationError("allowed_task_ids contains duplicates")
            unknown = set(allowed) - set(plan_view.remaining_task_ids)
            if unknown:
                raise CompilationError(
                    f"allowed_task_ids contains tasks outside active remaining order: {sorted(unknown)}"
                )
            allowed_set = set(allowed)
        else:
            allowed_set = None
        ready_ids = [
            task_id
            for task_id in plan_view.remaining_task_ids
            if self.runtime.facts(task_id).state == READY_UNCOMMITTED
            and (allowed_set is None or task_id in allowed_set)
        ]
        if not ready_ids:
            return CompileAttempt(code="NO_READY_TASK", batch=None, ready_task_count=0)

        selected_views: list[CanonicalTaskView] = []
        selected_ids: list[str] = []
        limit = max(1, int(self.resources.batch_limit(snapshot)))
        for task_id in ready_ids:
            candidate = self.catalogue.view(task_id)
            if self.resources.can_add(
                snapshot, selected_views, candidate, now_ns=int(now_ns)
            ):
                selected_views.append(candidate)
                selected_ids.append(task_id)
                if len(selected_ids) >= limit:
                    break
        if not selected_ids:
            return CompileAttempt(
                code="RETRYABLE_RESOURCE_BUSY",
                batch=None,
                ready_task_count=len(ready_ids),
            )

        authority_stamp = self.authority.authority_stamp(phase_key)
        topology_digest = self.resources.topology_digest(snapshot)
        link_class = self.resources.batch_link_class(snapshot, selected_views)
        batch_id = stable_id(
            "batch",
            {
                "phase": self.authority.adapter.phase_payload(phase_key),
                "task_ids": tuple(selected_ids),
                "authority_stamp": authority_stamp,
                "link_class": link_class,
                "topology_digest": topology_digest,
                "compiled_at_ns": int(now_ns),
            },
        )
        batch = make_transfer_batch(
            batch_id=batch_id,
            phase_key=phase_key,
            task_ids=tuple(selected_ids),
            authority_stamp=authority_stamp,
            link_class=link_class,
            topology_digest=topology_digest,
            compiled_at_ns=int(now_ns),
        )
        return CompileAttempt(
            code="BATCH_READY",
            batch=batch,
            ready_task_count=len(ready_ids),
        )


class BatchValidator:
    def __init__(
        self,
        *,
        catalogue: TaskCatalogue,
        runtime: TaskRuntimeIndex,
        authority: PhaseAuthorityManager,
        resources: BatchResourceAdapter | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.runtime = runtime
        self.authority = authority
        self.resources = resources or EndpointConflictResourceAdapter()

    def validate(
        self, batch: TransferBatch, *, snapshot: Any, now_ns: int
    ) -> ValidationResult:
        if not self.authority.stamp_is_current(batch.phase_key, batch.authority_stamp):
            return ValidationResult(
                code="RETRYABLE_STALE_AUTHORITY",
                reason="compiled batch authority no longer matches active plan epoch",
            )
        if str(batch.topology_digest) != str(self.resources.topology_digest(snapshot)):
            return ValidationResult(
                code="FATAL_TOPOLOGY_CONTRACT_MISMATCH",
                reason="batch and resource snapshot topology digests differ",
            )
        if not batch.task_ids:
            return ValidationResult(code="FATAL_CONTRACT_ERROR", reason="empty batch")
        if len(set(batch.task_ids)) != len(batch.task_ids):
            return ValidationResult(code="FATAL_CONTRACT_ERROR", reason="duplicate task in batch")
        phase_ids = set(self.catalogue.task_ids_for_phase(batch.phase_key))
        if any(task_id not in phase_ids for task_id in batch.task_ids):
            return ValidationResult(
                code="FATAL_CONTRACT_ERROR", reason="batch contains task from another phase"
            )
        if any(
            self.runtime.facts(task_id).state != READY_UNCOMMITTED
            for task_id in batch.task_ids
        ):
            return ValidationResult(
                code="FATAL_CONTRACT_ERROR", reason="batch task is not READY_UNCOMMITTED"
            )
        tasks = [self.catalogue.view(task_id) for task_id in batch.task_ids]
        if self.resources.batch_link_class(snapshot, tasks) is not batch.link_class:
            return ValidationResult(
                code="FATAL_TOPOLOGY_CONTRACT_MISMATCH",
                reason="batch link class differs from shared resolver",
            )
        if not self.resources.validate_batch(snapshot, tasks, now_ns=int(now_ns)):
            return ValidationResult(
                code="RETRYABLE_RESOURCE_BUSY", reason="resource snapshot changed or is busy"
            )
        return ValidationResult(code="ACCEPTED")


class ImmediateTransportPort(Protocol):
    def prepare_commit(
        self, batch: TransferBatch, commit_time_ns: int
    ) -> tuple[SubmitOutcome | str, CommitReceipt | None]: ...

    def confirm_commit(self, receipt: CommitReceipt) -> None: ...

    def abort_commit(self, receipt: CommitReceipt) -> None: ...


class ExecutionStabilizer:
    """Immediate-commit loop with no scheduler or transport waiting queue."""

    def __init__(
        self,
        *,
        compiler: BatchCompiler,
        validator: BatchValidator,
        authority: PhaseAuthorityManager,
        max_stale_retries: int = 32,
    ) -> None:
        self.compiler = compiler
        self.validator = validator
        self.authority = authority
        self.max_stale_retries = int(max_stale_retries)

    @staticmethod
    def _result_name(value: Any) -> str:
        name = getattr(value, "name", None)
        if isinstance(name, str):
            return name
        text = str(value)
        return text.rsplit(".", 1)[-1]

    def stabilize(
        self,
        *,
        phase_key: Any,
        snapshot_provider: Callable[[], Any],
        transport: ImmediateTransportPort,
        now_ns: int,
        allowed_task_ids: Sequence[str] | None = None,
    ) -> StabilizationResult:
        accepted_batches: list[str] = []
        accepted_tasks: list[str] = []
        stale_retries = 0
        remaining_allowed = (
            None
            if allowed_task_ids is None
            else tuple(dict.fromkeys(str(item) for item in allowed_task_ids))
        )
        while True:
            snapshot = snapshot_provider()
            attempt = self.compiler.compile_next(
                phase_key=phase_key,
                snapshot=snapshot,
                now_ns=int(now_ns),
                allowed_task_ids=remaining_allowed,
            )
            if attempt.code != "BATCH_READY" or attempt.batch is None:
                return StabilizationResult(
                    accepted_batch_digests=tuple(accepted_batches),
                    accepted_task_ids=tuple(accepted_tasks),
                    terminal_code=attempt.code,
                    stale_retries=stale_retries,
                )
            validation = self.validator.validate(
                attempt.batch, snapshot=snapshot, now_ns=int(now_ns)
            )
            if validation.code == "RETRYABLE_STALE_AUTHORITY":
                stale_retries += 1
                if stale_retries > self.max_stale_retries:
                    raise CompilationError("stale authority retry limit exceeded")
                continue
            if validation.code == "RETRYABLE_RESOURCE_BUSY":
                return StabilizationResult(
                    accepted_batch_digests=tuple(accepted_batches),
                    accepted_task_ids=tuple(accepted_tasks),
                    terminal_code=validation.code,
                    stale_retries=stale_retries,
                )
            if validation.code != "ACCEPTED":
                raise CompilationError(validation.reason or validation.code)

            raw_outcome, receipt = transport.prepare_commit(
                attempt.batch, commit_time_ns=int(now_ns)
            )
            submit_result = self._result_name(raw_outcome)
            if submit_result == "PREPARED":
                if receipt is None:
                    raise CompilationError("transport PREPARED without CommitReceipt")
                if not isinstance(receipt, CommitReceipt):
                    # A non-contract object cannot safely be confirmed.  The
                    # provider still owns any provisional resources and must
                    # accept its own returned handle for abort.
                    transport.abort_commit(receipt)
                    raise CompilationError("transport PREPARED with non-CommitReceipt object")
                # Formal transport must echo the exact immutable batch identity.
                # A digest-derived compatibility alias is not an authority and
                # is therefore rejected before Scheduler applies the receipt.
                mismatch = (
                    receipt.batch_id != attempt.batch.batch_id
                    or receipt.batch_digest != attempt.batch.batch_digest
                    or tuple(receipt.task_ids) != tuple(attempt.batch.task_ids)
                    or receipt.phase_key != phase_key
                    or receipt.authority_stamp != attempt.batch.authority_stamp
                    or receipt.topology_digest != attempt.batch.topology_digest
                    or receipt.commit_time_ns != int(now_ns)
                )
                if mismatch:
                    transport.abort_commit(receipt)
                    raise CompilationError("CommitReceipt does not echo the compiled batch contract")
                try:
                    self.authority.apply_commit_receipt(receipt)
                except Exception:
                    transport.abort_commit(receipt)
                    raise
                # Required contract: valid live receipt confirmation is infallible.
                transport.confirm_commit(receipt)
                accepted_batches.append(attempt.batch.batch_digest)
                accepted_tasks.extend(attempt.batch.task_ids)
                if remaining_allowed is not None:
                    committed = set(attempt.batch.task_ids)
                    remaining_allowed = tuple(
                        task_id for task_id in remaining_allowed if task_id not in committed
                    )
                    if not remaining_allowed:
                        return StabilizationResult(
                            accepted_batch_digests=tuple(accepted_batches),
                            accepted_task_ids=tuple(accepted_tasks),
                            terminal_code="ALLOWED_PREFIX_COMMITTED",
                            stale_retries=stale_retries,
                        )
                continue
            if receipt is not None:
                transport.abort_commit(receipt)
                raise CompilationError("non-PREPARED transport result carried a receipt")
            if submit_result == "RETRYABLE_STALE_AUTHORITY":
                stale_retries += 1
                if stale_retries > self.max_stale_retries:
                    raise CompilationError("transport stale authority retry limit exceeded")
                continue
            if submit_result == "RETRYABLE_RESOURCE_BUSY":
                return StabilizationResult(
                    accepted_batch_digests=tuple(accepted_batches),
                    accepted_task_ids=tuple(accepted_tasks),
                    terminal_code=submit_result,
                    stale_retries=stale_retries,
                )
            raise CompilationError(f"transport returned fatal result {submit_result}")
