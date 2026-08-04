from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from rs_sim.contracts.factories import (
    ceil_transfer_time_ns,
    hardware_profile_digest,
    make_commit_receipt,
    make_transport_snapshot,
)
from rs_sim.backend.core.simulation import ProgressSignal, SimulationKernel
from rs_sim.contracts.schema import (
    CanonicalTransferTask,
    CommitReceipt,
    HardwareProfile,
    KernelPhase,
    LinkClass,
    PhysicalTransferRecord,
    ReceivePermit,
    SimulationEvent,
    SubmitOutcome,
    TaskResourceFootprint,
    TransferBatch,
    TransferCompleted,
    TransferStarted,
    TransportSnapshot,
    PhaseKey,
    WindowKey,
)
from rs_sim.contracts.digest import stable_digest
from rs_sim.transport.api.ports import (
    AuthorityValidationPort,
    CompletionSink,
    PermitLookupPort,
    ResourceReleaseSink,
    TaskLookupPort,
    TaskResourceResolverPort,
)

from ..core.ordering import semantic_ordinal
from ..core.errors import ReceiptStateError, RejectionCode, TransportRejection
from ..core.lifecycle import make_process_lifecycle_diagnostics
from ..observability.metrics import (
    PhysicalBusyInterval,
    PhysicalLaunchMetric,
    PhysicalMetricsView,
    PhysicalTaskMetric,
)
from ..config.profiles import (
    BANDWIDTH_MODE_FIXED_PER_LANE,
    BandwidthContentionSensitivity,
    fixed_per_lane_bandwidth_sensitivity,
)


@dataclass(frozen=True, slots=True)
class _ProvisionalTaskReservation:
    task: CanonicalTransferTask
    permit: ReceivePermit
    footprint: TaskResourceFootprint
    lane_id: str
    committed_at_ns: int
    start_at_ns: int
    complete_at_ns: int


@dataclass(frozen=True, slots=True)
class _ActiveTransfer:
    task: CanonicalTransferTask
    footprint: TaskResourceFootprint
    record: PhysicalTransferRecord


@dataclass(frozen=True, slots=True)
class _DataMechanismEvent:
    phase_key: PhaseKey
    kind: str
    link_class: LinkClass | None
    task_count: int
    payload_bytes: int
    duration_ns: int
    at_ns: int


@dataclass(frozen=True, slots=True)
class _PreparedCommit:
    receipt: CommitReceipt
    batch: TransferBatch
    reserved: tuple[_ProvisionalTaskReservation, ...]
    contention_by_task: tuple[tuple[str, str, int, int, int], ...]
    prepared_at_kernel_ns: int


class FormalDataPlaneTransport:
    """Transaction-based physical data plane.

    This class owns physical records and resource occupancy only.  It does not
    own canonical logical TaskState and has no policy waiting queue.
    """

    START_EVENT = "TRANSPORT_TRANSFER_START"
    COMPLETE_EVENT = "TRANSPORT_TRANSFER_COMPLETE"
    PRODUCER = "RS_SIM_TRANSPORT_DATA_PLANE"

    def __init__(
        self,
        *,
        kernel: SimulationKernel,
        task_lookup: TaskLookupPort,
        permit_lookup: PermitLookupPort,
        authority_validation: AuthorityValidationPort,
        resource_resolver: TaskResourceResolverPort,
        completion_sink: CompletionSink,
        resource_release_sink: ResourceReleaseSink,
        hardware_profile: HardwareProfile,
        bandwidth_contention: BandwidthContentionSensitivity | None = None,
    ) -> None:
        if not isinstance(kernel, SimulationKernel):
            raise TypeError("kernel must be SimulationKernel")
        if not isinstance(hardware_profile, HardwareProfile):
            raise TypeError("hardware_profile must be HardwareProfile")
        if hardware_profile_digest(hardware_profile) != hardware_profile.profile_digest:
            raise ValueError("hardware profile digest does not match its semantic fields")

        topology = resource_resolver.topology
        profile_classes = {
            link_class
            for link_class, _ in hardware_profile.bandwidth_bytes_per_second_by_link_class
        }
        topology_classes = {
            link_class for link_class, lanes in topology.lane_ids_by_link_class if lanes
        }
        if not topology_classes.issubset(profile_classes):
            missing = sorted(item.value for item in topology_classes - profile_classes)
            raise ValueError(f"hardware profile missing link classes: {missing}")
        launch_classes = {item for item, _ in hardware_profile.launch_delay_ns_by_link_class}
        fixed_classes = {item for item, _ in hardware_profile.fixed_latency_ns_by_link_class}
        if not topology_classes.issubset(launch_classes & fixed_classes):
            raise ValueError("hardware profile latency maps do not cover topology link classes")

        self.kernel = kernel
        self.task_lookup = task_lookup
        self.permit_lookup = permit_lookup
        self.authority_validation = authority_validation
        self.resource_resolver = resource_resolver
        self.completion_sink = completion_sink
        self.resource_release_sink = resource_release_sink
        self.hardware_profile = hardware_profile
        self._topology = topology
        self.bandwidth_contention = (
            bandwidth_contention or fixed_per_lane_bandwidth_sensitivity()
        )
        if not isinstance(
            self.bandwidth_contention, BandwidthContentionSensitivity
        ):
            raise TypeError(
                "bandwidth_contention must be BandwidthContentionSensitivity"
            )

        self._busy_src: set[int] = set()
        self._busy_dst: set[int] = set()
        self._busy_nics: set[str] = set()
        self._busy_lanes: set[str] = set()

        self._prepared: dict[str, _PreparedCommit] = {}
        self._aborted_receipt_ids: set[str] = set()
        self._confirmed_receipt_ids: set[str] = set()
        self._known_physical_task_ids: set[str] = set()
        self._active_transfers: dict[str, _ActiveTransfer] = {}
        self._records_by_task: dict[str, PhysicalTransferRecord] = {}
        self._completed_task_ids: set[str] = set()
        self._event_record_by_id: dict[str, PhysicalTransferRecord] = {}
        self._last_rejection: TransportRejection | None = None

        # Authoritative integer-only evidence. These counters describe the
        # transport mechanism; they do not imply calibrated performance.
        self._prepare_attempt_count = 0
        self._prepared_commit_count_total = 0
        self._confirmed_commit_count = 0
        self._aborted_commit_count = 0
        self._rejection_counts: dict[str, int] = {}
        self._resource_busy_retry_counts: dict[str, int] = {}
        self._launch_count_by_link_class: dict[LinkClass, int] = {}
        self._launch_cost_ns_by_link_class: dict[LinkClass, int] = {}
        self._committed_bytes_by_link_class: dict[LinkClass, int] = {}
        self._completed_bytes_by_link_class: dict[LinkClass, int] = {}
        self._transfer_busy_time_ns_by_link_class: dict[LinkClass, int] = {}
        self._batch_span_busy_time_ns_by_link_class: dict[LinkClass, int] = {}
        self._lane_busy_time_ns: dict[str, int] = {
            lane_id: 0
            for _, lane_ids in topology.lane_ids_by_link_class
            for lane_id in lane_ids
        }
        self._nic_busy_time_ns: dict[str, int] = {
            nic_id: 0
            for nic_id in (*topology.tx_nic_id_by_rank, *topology.rx_nic_id_by_rank)
        }
        self._rank_tx_busy_time_ns: dict[int, int] = {
            rank: 0 for rank in range(topology.world_size)
        }
        self._rank_rx_busy_time_ns: dict[int, int] = {
            rank: 0 for rank in range(topology.world_size)
        }
        self._first_physical_start_ns: int | None = None
        self._last_physical_completion_ns: int | None = None
        self._mechanism_events: list[_DataMechanismEvent] = []
        self._started_task_ids: set[str] = set()
        self._physical_task_by_id: dict[str, CanonicalTransferTask] = {}
        self._resource_footprint_by_task: dict[str, TaskResourceFootprint] = {}
        self._confirmed_receipt_task_ids: dict[str, tuple[str, ...]] = {}
        self._contention_detail_by_task: dict[str, tuple[str, int, int, int]] = {}
        self._contention_affected_task_count = 0
        self._contention_extra_service_ns = 0
        self._max_contention_share_count = 1
        self._closed = False
        self._disposed = False
        self._kernel_callbacks_disposed = False
        self._final_lifecycle_evidence_digest: str | None = None
        self._closed_lifecycle_diagnostics: dict[str, Any] | None = None
        self._disposed_lifecycle_diagnostics: dict[str, Any] | None = None

        kernel.register_event_handler(self.START_EVENT, self._handle_start)
        kernel.register_event_handler(self.COMPLETE_EVENT, self._handle_complete)
        kernel.register_evidence_provider("transport_data_plane", self.evidence)

    @property
    def topology(self):
        return self._topology

    def _ensure_open(self) -> None:
        if self._closed or self._disposed:
            raise RuntimeError("Formal transport DataPlane is closed")

    @staticmethod
    def _map(rows: Iterable[tuple[LinkClass, int]]) -> dict[LinkClass, int]:
        return dict(rows)

    @property
    def last_rejection(self) -> TransportRejection | None:
        return self._last_rejection

    @property
    def prepared_count(self) -> int:
        return len(self._prepared)

    @property
    def committed_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._known_physical_task_ids - self._completed_task_ids))

    @property
    def completed_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._completed_task_ids))

    def physical_records(self) -> tuple[PhysicalTransferRecord, ...]:
        return tuple(self._records_by_task[key] for key in sorted(self._records_by_task))

    def physical_record_digest(self) -> str:
        return stable_digest(self.physical_records(), domain="TRANSPORT_PHYSICAL_RECORDS")

    @staticmethod
    def _normalize_phase_filter(
        phase_keys: Iterable[PhaseKey],
    ) -> tuple[PhaseKey, ...]:
        values = tuple(phase_keys)
        if not all(isinstance(item, PhaseKey) for item in values):
            raise TypeError("phase_keys must contain only PhaseKey values")
        return tuple(sorted(set(values), key=FormalDataPlaneTransport._phase_sort_key))

    @staticmethod
    def _normalize_task_filter(task_ids: Iterable[str], *, name: str) -> tuple[str, ...]:
        values = tuple(task_ids)
        if not all(isinstance(item, str) and item for item in values):
            raise TypeError(f"{name} must contain only non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
        return tuple(sorted(values))

    def _outstanding_confirmed_receipt_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                receipt_id
                for receipt_id, task_ids in self._confirmed_receipt_task_ids.items()
                if not set(task_ids).issubset(self._completed_task_ids)
            )
        )

    def physical_metrics(
        self,
        *,
        phase_keys: Iterable[PhaseKey] = (),
        task_ids: Iterable[str] = (),
        window_task_ids: Iterable[str] = (),
    ) -> PhysicalMetricsView:
        """Return an immutable exact physical view for anchor/window accounting.

        Filters are intersected. ``window_task_ids`` is intentionally supplied by
        the Integration Owner because overlapping PlanningWindows do not own
        physical state and cannot be inferred from a single phase-to-window map.
        Unknown task IDs fail closed rather than silently producing partial sums.
        """

        phases = self._normalize_phase_filter(phase_keys)
        requested_tasks = self._normalize_task_filter(task_ids, name="task_ids")
        requested_window_tasks = self._normalize_task_filter(
            window_task_ids, name="window_task_ids"
        )

        prepared_task_by_id: dict[str, CanonicalTransferTask] = {}
        for prepared in self._prepared.values():
            for item in prepared.reserved:
                prepared_task_by_id[item.task.task_id] = item.task
        all_task_by_id = {**prepared_task_by_id, **self._physical_task_by_id}
        known_task_ids = set(all_task_by_id)
        requested_union = set(requested_tasks) | set(requested_window_tasks)
        unknown = sorted(requested_union - known_task_ids)
        if unknown:
            raise KeyError(f"unknown transport physical metric task IDs: {unknown}")

        selected = set(known_task_ids)
        if phases:
            selected &= {
                task_id
                for task_id, task in all_task_by_id.items()
                if task.phase_key in set(phases)
            }
        if requested_tasks:
            selected &= set(requested_tasks)
        if requested_window_tasks:
            selected &= set(requested_window_tasks)
        selected_task_ids = tuple(sorted(selected))
        filter_active = bool(phases or requested_tasks or requested_window_tasks)

        task_metrics: list[PhysicalTaskMetric] = []
        for task_id in selected_task_ids:
            record = self._records_by_task.get(task_id)
            if record is None:
                continue
            task = self._physical_task_by_id[task_id]
            footprint = self._resource_footprint_by_task[task_id]
            task_metrics.append(
                PhysicalTaskMetric(
                    task_id=task_id,
                    phase_key=task.phase_key,
                    batch_id=record.batch_id,
                    link_class=record.link_class,
                    lane_id=record.lane_id,
                    tx_nic_id=footprint.tx_nic_id,
                    rx_nic_id=footprint.rx_nic_id,
                    committed_at_ns=record.committed_at_ns,
                    start_at_ns=record.start_at_ns,
                    complete_at_ns=record.complete_at_ns,
                    payload_bytes=record.payload_bytes,
                    completed=task_id in self._completed_task_ids,
                )
            )

        records_by_batch: dict[str, list[PhysicalTaskMetric]] = {}
        for metric in task_metrics:
            records_by_batch.setdefault(metric.batch_id, []).append(metric)
        launches: list[PhysicalLaunchMetric] = []
        for batch_id in sorted(records_by_batch):
            selected_rows = records_by_batch[batch_id]
            all_rows = tuple(
                sorted(
                    (
                        task_id,
                        self._records_by_task[task_id],
                        self._physical_task_by_id[task_id],
                    )
                    for task_id in self._records_by_task
                    if self._records_by_task[task_id].batch_id == batch_id
                )
            )
            first = selected_rows[0]
            launches.append(
                PhysicalLaunchMetric(
                    batch_id=batch_id,
                    phase_key=first.phase_key,
                    link_class=first.link_class,
                    physical_batch_task_ids=tuple(row[0] for row in all_rows),
                    selected_task_ids=tuple(sorted(row.task_id for row in selected_rows)),
                    committed_at_ns=min(row[1].committed_at_ns for row in all_rows),
                    start_at_ns=min(row[1].start_at_ns for row in all_rows),
                    complete_at_ns=max(row[1].complete_at_ns for row in all_rows),
                    launch_delay_ns=(
                        min(row[1].start_at_ns for row in all_rows)
                        - min(row[1].committed_at_ns for row in all_rows)
                    ),
                )
            )

        intervals: list[PhysicalBusyInterval] = []
        for metric in task_metrics:
            resources = (
                ("LINK_CLASS", metric.link_class.value),
                ("LANE", metric.lane_id),
                *(("NIC", nic_id) for nic_id in sorted({metric.tx_nic_id, metric.rx_nic_id})),
            )
            for resource_kind, resource_id in resources:
                intervals.append(
                    PhysicalBusyInterval(
                        resource_kind=resource_kind,
                        resource_id=resource_id,
                        phase_key=metric.phase_key,
                        task_id=metric.task_id,
                        batch_id=metric.batch_id,
                        start_at_ns=metric.start_at_ns,
                        complete_at_ns=metric.complete_at_ns,
                    )
                )
        intervals.sort(
            key=lambda item: (
                item.resource_kind,
                item.resource_id,
                item.start_at_ns,
                item.complete_at_ns,
                item.task_id,
            )
        )

        prepared_receipts = tuple(
            sorted(
                receipt_id
                for receipt_id, prepared in self._prepared.items()
                if not filter_active or set(prepared.batch.task_ids) & selected
            )
        )
        outstanding_confirmed = tuple(
            receipt_id
            for receipt_id in self._outstanding_confirmed_receipt_ids()
            if not filter_active
            or set(self._confirmed_receipt_task_ids[receipt_id]) & selected
        )
        terminal = self.terminal_state()
        completed_bytes = sum(
            metric.payload_bytes for metric in task_metrics if metric.completed
        )
        return PhysicalMetricsView(
            requested_phase_keys=phases,
            requested_task_ids=requested_tasks,
            requested_window_task_ids=requested_window_tasks,
            selected_task_ids=selected_task_ids,
            task_metrics=tuple(task_metrics),
            launch_metrics=tuple(launches),
            busy_intervals=tuple(intervals),
            outstanding_prepared_receipt_ids=prepared_receipts,
            outstanding_confirmed_receipt_ids=outstanding_confirmed,
            physical_completed_bytes=completed_bytes,
            launch_count=len(launches),
            launch_delay_total_ns=sum(item.launch_delay_ns for item in launches),
            all_resources_free=bool(terminal["all_resources_free"]),
            terminal=bool(terminal["terminal"]),
        )

    @staticmethod
    def _bump(counter: dict[Any, int], key: Any, amount: int = 1) -> None:
        counter[key] = int(counter.get(key, 0)) + int(amount)

    @staticmethod
    def _window_for_phase(phase_key: PhaseKey) -> WindowKey:
        return WindowKey(
            run_id=phase_key.run_id,
            sample_id=phase_key.sample_id,
            window_index=phase_key.layer_index,
        )

    @staticmethod
    def _phase_sort_key(phase_key: PhaseKey) -> tuple[str, str, int, str]:
        return (
            phase_key.run_id,
            phase_key.sample_id,
            phase_key.layer_index,
            phase_key.phase_kind.value,
        )

    @staticmethod
    def _window_sort_key(window_key: WindowKey) -> tuple[str, str, int]:
        return (window_key.run_id, window_key.sample_id, window_key.window_index)

    def _record_mechanism_event(
        self,
        *,
        phase_key: PhaseKey,
        kind: str,
        link_class: LinkClass | None = None,
        task_count: int = 0,
        payload_bytes: int = 0,
        duration_ns: int = 0,
        at_ns: int | None = None,
    ) -> None:
        self._mechanism_events.append(
            _DataMechanismEvent(
                phase_key=phase_key,
                kind=str(kind),
                link_class=link_class,
                task_count=int(task_count),
                payload_bytes=int(payload_bytes),
                duration_ns=int(duration_ns),
                at_ns=int(self.kernel.now_ns if at_ns is None else at_ns),
            )
        )

    def _scoped_statistics(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        phase_rows: dict[PhaseKey, dict[str, Any]] = {}
        window_rows: dict[WindowKey, dict[str, Any]] = {}

        def bump(values: dict[str, int], key: str, amount: int = 1) -> None:
            values[key] = int(values.get(key, 0)) + int(amount)

        def update(scope: dict[str, Any], event: _DataMechanismEvent) -> None:
            counters = scope.setdefault("counters", {})
            bump(counters, "event_count")
            if event.kind == "PREPARE_ATTEMPT":
                bump(counters, "prepare_attempt_count")
            elif event.kind == "PREPARED":
                bump(counters, "prepared_commit_count")
                bump(counters, "prepared_task_count", event.task_count)
                bump(counters, "prepared_bytes", event.payload_bytes)
            elif event.kind == "CONFIRMED":
                bump(counters, "confirmed_commit_count")
                bump(counters, "launch_count")
                bump(counters, "launch_cost_ns", event.duration_ns)
                bump(counters, "committed_task_count", event.task_count)
                bump(counters, "committed_bytes", event.payload_bytes)
            elif event.kind == "ABORTED":
                bump(counters, "aborted_commit_count")
                bump(counters, "aborted_task_count", event.task_count)
                bump(counters, "aborted_bytes", event.payload_bytes)
            elif event.kind == "STARTED":
                bump(counters, "started_task_count", event.task_count)
                bump(counters, "started_bytes", event.payload_bytes)
            elif event.kind == "COMPLETED":
                bump(counters, "completed_task_count", event.task_count)
                bump(counters, "completed_bytes", event.payload_bytes)
                bump(counters, "transfer_busy_time_ns", event.duration_ns)
            elif event.kind.startswith("REJECT_"):
                bump(counters, "rejection_count")
                bump(counters, f"{event.kind.lower()}_count")
                if event.kind == "REJECT_RESOURCE_BUSY":
                    bump(counters, "resource_busy_retry_count")

            scope["first_event_ns"] = min(
                int(scope.get("first_event_ns", event.at_ns)), event.at_ns
            )
            scope["last_event_ns"] = max(
                int(scope.get("last_event_ns", event.at_ns)), event.at_ns
            )
            if event.link_class is not None:
                link_rows = scope.setdefault("link_rows", {})
                link = link_rows.setdefault(event.link_class, {})
                bump(link, "event_count")
                if event.kind == "CONFIRMED":
                    bump(link, "launch_count")
                    bump(link, "launch_cost_ns", event.duration_ns)
                    bump(link, "committed_task_count", event.task_count)
                    bump(link, "committed_bytes", event.payload_bytes)
                elif event.kind == "COMPLETED":
                    bump(link, "completed_task_count", event.task_count)
                    bump(link, "completed_bytes", event.payload_bytes)
                    bump(link, "transfer_busy_time_ns", event.duration_ns)
                elif event.kind == "REJECT_RESOURCE_BUSY":
                    bump(link, "resource_busy_retry_count")

        for event in self._mechanism_events:
            update(phase_rows.setdefault(event.phase_key, {}), event)
            update(
                window_rows.setdefault(self._window_for_phase(event.phase_key), {}),
                event,
            )

        def freeze(key: Any, scope: dict[str, Any]) -> tuple[Any, ...]:
            link_rows = scope.get("link_rows", {})
            return (
                key,
                int(scope.get("first_event_ns", self.kernel.now_ns)),
                int(scope.get("last_event_ns", self.kernel.now_ns)),
                tuple(sorted(scope.get("counters", {}).items())),
                tuple(
                    (link_class.value, tuple(sorted(values.items())))
                    for link_class, values in sorted(
                        link_rows.items(), key=lambda item: item[0].value
                    )
                ),
            )

        phases = tuple(
            freeze(key, phase_rows[key])
            for key in sorted(phase_rows, key=self._phase_sort_key)
        )
        windows = tuple(
            freeze(key, window_rows[key])
            for key in sorted(window_rows, key=self._window_sort_key)
        )
        return phases, windows

    def _observation_window(self) -> tuple[int, int, int]:
        if self._first_physical_start_ns is None:
            now = int(self.kernel.now_ns)
            return now, now, 0
        start = int(self._first_physical_start_ns)
        end = max(
            int(self.kernel.now_ns),
            int(self._last_physical_completion_ns or start),
        )
        return start, end, max(0, end - start)

    def _completed_record_rows(self) -> tuple[tuple[Any, ...], ...]:
        rows: list[tuple[Any, ...]] = []
        for task_id in sorted(self._completed_task_ids):
            record = self._records_by_task[task_id]
            task = self._physical_task_by_id[task_id]
            group_id, share_count, base_transfer_ns, effective_transfer_ns = (
                self._contention_detail_by_task[task_id]
            )
            rows.append(
                (
                    task_id,
                    task.phase_key,
                    record.batch_id,
                    record.link_class.value,
                    record.lane_id,
                    record.committed_at_ns,
                    record.start_at_ns,
                    record.complete_at_ns,
                    record.payload_bytes,
                    group_id,
                    share_count,
                    base_transfer_ns,
                    effective_transfer_ns,
                )
            )
        return tuple(rows)

    def _statistics_reconciliation(self) -> dict[str, Any]:
        completed_records = tuple(
            self._records_by_task[task_id]
            for task_id in sorted(self._completed_task_ids)
        )
        bytes_by_link: dict[LinkClass, int] = {}
        for record in completed_records:
            self._bump(bytes_by_link, record.link_class, record.payload_bytes)
        record_rows = tuple(
            (link_class.value, value)
            for link_class, value in sorted(
                bytes_by_link.items(), key=lambda item: item[0].value
            )
        )
        counter_rows = tuple(
            (link_class.value, value)
            for link_class, value in sorted(
                self._completed_bytes_by_link_class.items(),
                key=lambda item: item[0].value,
            )
        )
        record_bytes = sum(record.payload_bytes for record in completed_records)
        counter_bytes = sum(self._completed_bytes_by_link_class.values())
        reconciled = bool(
            record_rows == counter_rows
            and record_bytes == counter_bytes
            and len(completed_records) == len(self._completed_task_ids)
            and self._completed_task_ids.issubset(self._records_by_task)
        )
        return {
            "reconciled": reconciled,
            "completed_record_count": len(completed_records),
            "completed_record_bytes": record_bytes,
            "counter_completed_bytes": counter_bytes,
            "completed_record_bytes_by_link_class": record_rows,
            "counter_completed_bytes_by_link_class": counter_rows,
        }

    def assert_statistics_reconcile(self) -> None:
        reconciliation = self._statistics_reconciliation()
        if not reconciliation["reconciled"]:
            raise RuntimeError(
                f"Formal transport DataPlane statistics do not reconcile: {reconciliation}"
            )

    def statistics(self) -> dict[str, Any]:
        start_ns, end_ns, window_ns = self._observation_window()
        phase_statistics, window_statistics = self._scoped_statistics()
        link_classes = sorted(
            {
                *self._launch_count_by_link_class,
                *self._launch_cost_ns_by_link_class,
                *self._committed_bytes_by_link_class,
                *self._completed_bytes_by_link_class,
                *self._transfer_busy_time_ns_by_link_class,
                *self._batch_span_busy_time_ns_by_link_class,
            },
            key=lambda item: item.value,
        )
        reconciliation = self._statistics_reconciliation()
        payload: dict[str, Any] = {
            "schema_version": "TRANSPORT_DATA_PLANE_STATISTICS",
            "profile_provenance": self.hardware_profile.profile_provenance,
            "performance_eligible": self.hardware_profile.performance_eligible,
            "observation_start_ns": start_ns,
            "observation_end_ns": end_ns,
            "observation_window_ns": window_ns,
            "prepare_attempt_count": self._prepare_attempt_count,
            "prepared_commit_count": self._prepared_commit_count_total,
            "confirmed_commit_count": self._confirmed_commit_count,
            "aborted_commit_count": self._aborted_commit_count,
            "rejection_counts": tuple(sorted(self._rejection_counts.items())),
            "resource_wait_retry_counts": tuple(
                sorted(self._resource_busy_retry_counts.items())
            ),
            "total_launch_count": sum(self._launch_count_by_link_class.values()),
            "total_launch_cost_ns": sum(
                self._launch_cost_ns_by_link_class.values()
            ),
            "link_class_statistics": tuple(
                (
                    link_class.value,
                    (
                        (
                            "launch_count",
                            self._launch_count_by_link_class.get(link_class, 0),
                        ),
                        (
                            "launch_cost_ns",
                            self._launch_cost_ns_by_link_class.get(link_class, 0),
                        ),
                        (
                            "committed_bytes",
                            self._committed_bytes_by_link_class.get(link_class, 0),
                        ),
                        (
                            "completed_bytes",
                            self._completed_bytes_by_link_class.get(link_class, 0),
                        ),
                        (
                            "transfer_busy_time_ns",
                            self._transfer_busy_time_ns_by_link_class.get(
                                link_class, 0
                            ),
                        ),
                        (
                            "batch_span_busy_time_ns",
                            self._batch_span_busy_time_ns_by_link_class.get(
                                link_class, 0
                            ),
                        ),
                    ),
                )
                for link_class in link_classes
            ),
            "lane_busy_time_ns": tuple(sorted(self._lane_busy_time_ns.items())),
            "nic_busy_time_ns": tuple(sorted(self._nic_busy_time_ns.items())),
            "rank_tx_busy_time_ns": tuple(
                sorted(self._rank_tx_busy_time_ns.items())
            ),
            "rank_rx_busy_time_ns": tuple(
                sorted(self._rank_rx_busy_time_ns.items())
            ),
            "lane_utilization_rational": tuple(
                (lane_id, busy_ns, window_ns)
                for lane_id, busy_ns in sorted(self._lane_busy_time_ns.items())
            ),
            "nic_utilization_rational": tuple(
                (nic_id, busy_ns, window_ns)
                for nic_id, busy_ns in sorted(self._nic_busy_time_ns.items())
            ),
            "rank_tx_utilization_rational": tuple(
                (rank, busy_ns, window_ns)
                for rank, busy_ns in sorted(self._rank_tx_busy_time_ns.items())
            ),
            "rank_rx_utilization_rational": tuple(
                (rank, busy_ns, window_ns)
                for rank, busy_ns in sorted(self._rank_rx_busy_time_ns.items())
            ),
            "physical_started_record_count": len(self._started_task_ids),
            "physical_completed_record_count": len(self._completed_task_ids),
            "physical_completed_bytes": sum(
                self._completed_bytes_by_link_class.values()
            ),
            "transfer_timeline": self._completed_record_rows(),
            "statistics_reconciliation": reconciliation,
            "bandwidth_contention_mode": self.bandwidth_contention.mode,
            "bandwidth_contention_config_digest": (
                self.bandwidth_contention.config_digest
            ),
            "contention_affected_task_count": (
                self._contention_affected_task_count
            ),
            "contention_extra_service_ns": self._contention_extra_service_ns,
            "max_contention_share_count": self._max_contention_share_count,
            "phase_mechanism_statistics": phase_statistics,
            "window_mechanism_statistics": window_statistics,
            "mechanism_statistics_digest": stable_digest(
                (phase_statistics, window_statistics),
                domain="TRANSPORT_DATA_SCOPED_STATISTICS",
            ),
        }
        payload["statistics_digest"] = stable_digest(
            payload, domain="TRANSPORT_DATA_PLANE_STATISTICS"
        )
        return payload

    def formal_runtime_metrics(self) -> dict[str, Any]:
        statistics = self.statistics()
        payload: dict[str, Any] = {
            "schema_version": "TRANSPORT_DATA_RUNTIME_METRICS",
            "hardware_profile_id": self.hardware_profile.profile_id,
            "hardware_profile_digest": self.hardware_profile.profile_digest,
            "profile_provenance": self.hardware_profile.profile_provenance,
            "performance_eligible": self.hardware_profile.performance_eligible,
            "topology_id": self.topology.topology_id,
            "topology_digest": self.topology.topology_digest,
            "statistics": statistics,
            "transfer_timeline": self._completed_record_rows(),
            "physical_record_digest": self.physical_record_digest(),
            "terminal_resource_evidence": self.terminal_state(),
            "internal_policy_wait_queue": False,
            "retained_rejected_batches": (),
            **self.bandwidth_contention.manifest_fragment(),
        }
        payload["runtime_metrics_digest"] = stable_digest(
            payload, domain="TRANSPORT_DATA_RUNTIME_METRICS"
        )
        return payload

    def terminal_state(self) -> dict[str, Any]:
        resources_free = not (
            self._busy_src or self._busy_dst or self._busy_nics or self._busy_lanes
        )
        pending_data_event_count = len(self._event_record_by_id)
        outstanding_confirmed_receipts = self._outstanding_confirmed_receipt_ids()
        terminal = bool(
            not self._prepared
            and not self._active_transfers
            and not outstanding_confirmed_receipts
            and resources_free
            and pending_data_event_count == 0
        )
        return {
            "terminal": terminal,
            "closed": self._closed,
            "disposed": self._disposed,
            "prepared_count": len(self._prepared),
            "outstanding_confirmed_receipt_count": len(
                outstanding_confirmed_receipts
            ),
            "active_transfer_count": len(self._active_transfers),
            "pending_data_event_count": pending_data_event_count,
            "all_resources_free": resources_free,
            "busy_src_rank_count": len(self._busy_src),
            "busy_dst_rank_count": len(self._busy_dst),
            "busy_nic_count": len(self._busy_nics),
            "busy_lane_count": len(self._busy_lanes),
            "physical_completed_bytes": sum(
                self._completed_bytes_by_link_class.values()
            ),
            "internal_wait_queue_depth": 0,
            "retained_rejected_batch_count": 0,
            "started_task_count": len(self._started_task_ids),
            "completed_task_count": len(self._completed_task_ids),
        }

    def assert_terminal(self) -> None:
        state = self.terminal_state()
        if not state["terminal"]:
            raise RuntimeError(f"Formal transport DataPlane is not terminal: {state}")

    def _build_lifecycle_diagnostics(self) -> dict[str, Any]:
        state = self.terminal_state()
        return make_process_lifecycle_diagnostics(
            component="TRANSPORT_DATA_PLANE",
            closed=self._closed,
            disposed=self._disposed,
            kernel_pending_event_count=int(self.kernel.pending_event_count()),
            live_receipt_count=(
                len(self._prepared)
                + len(self._outstanding_confirmed_receipt_ids())
            ),
            live_transfer_or_request_count=len(self._active_transfers),
            all_resources_free=bool(state["all_resources_free"]),
            final_evidence_digest=self._final_lifecycle_evidence_digest,
            kernel_callback_registry_disposed=self._kernel_callbacks_disposed,
        )

    def lifecycle_diagnostics(self) -> dict[str, Any]:
        if self._disposed:
            if self._disposed_lifecycle_diagnostics is None:
                self._disposed_lifecycle_diagnostics = (
                    self._build_lifecycle_diagnostics()
                )
            return dict(self._disposed_lifecycle_diagnostics)
        if self._closed:
            if self._closed_lifecycle_diagnostics is None:
                self._closed_lifecycle_diagnostics = (
                    self._build_lifecycle_diagnostics()
                )
            return dict(self._closed_lifecycle_diagnostics)
        return self._build_lifecycle_diagnostics()

    def close(self) -> dict[str, Any]:
        """Freeze terminal evidence. Safe to call repeatedly after a valid run."""

        if not self._closed:
            self.assert_terminal()
            self.assert_statistics_reconcile()
            self._final_lifecycle_evidence_digest = stable_digest(
                (
                    self.physical_records(),
                    self.statistics(),
                    self.terminal_state(),
                    tuple(sorted(self._confirmed_receipt_ids)),
                    tuple(sorted(self._aborted_receipt_ids)),
                ),
                domain="TRANSPORT_DATA_PLANE_FINAL_LIFECYCLE_EVIDENCE",
            )
            self._closed = True
            self._closed_lifecycle_diagnostics = (
                self._build_lifecycle_diagnostics()
            )
        return self.lifecycle_diagnostics()

    def _reset_to_disposed_baseline(self) -> None:
        self._busy_src.clear()
        self._busy_dst.clear()
        self._busy_nics.clear()
        self._busy_lanes.clear()
        self._prepared.clear()
        self._aborted_receipt_ids.clear()
        self._confirmed_receipt_ids.clear()
        self._confirmed_receipt_task_ids.clear()
        self._known_physical_task_ids.clear()
        self._active_transfers.clear()
        self._records_by_task.clear()
        self._completed_task_ids.clear()
        self._event_record_by_id.clear()
        self._started_task_ids.clear()
        self._physical_task_by_id.clear()
        self._resource_footprint_by_task.clear()
        self._contention_detail_by_task.clear()
        self._mechanism_events.clear()
        self._last_rejection = None
        self._prepare_attempt_count = 0
        self._prepared_commit_count_total = 0
        self._confirmed_commit_count = 0
        self._aborted_commit_count = 0
        self._rejection_counts.clear()
        self._resource_busy_retry_counts.clear()
        self._launch_count_by_link_class.clear()
        self._launch_cost_ns_by_link_class.clear()
        self._committed_bytes_by_link_class.clear()
        self._completed_bytes_by_link_class.clear()
        self._transfer_busy_time_ns_by_link_class.clear()
        self._batch_span_busy_time_ns_by_link_class.clear()
        for lane_id in tuple(self._lane_busy_time_ns):
            self._lane_busy_time_ns[lane_id] = 0
        for nic_id in tuple(self._nic_busy_time_ns):
            self._nic_busy_time_ns[nic_id] = 0
        for rank in tuple(self._rank_tx_busy_time_ns):
            self._rank_tx_busy_time_ns[rank] = 0
        for rank in tuple(self._rank_rx_busy_time_ns):
            self._rank_rx_busy_time_ns[rank] = 0
        self._first_physical_start_ns = None
        self._last_physical_completion_ns = None
        self._contention_affected_task_count = 0
        self._contention_extra_service_ns = 0
        self._max_contention_share_count = 1

    def _mark_kernel_callbacks_disposed(self) -> None:
        self._kernel_callbacks_disposed = True
        self._disposed_lifecycle_diagnostics = None

    def dispose(self, *, dispose_kernel: bool = False) -> dict[str, Any]:
        """Detach all owned references and reset counters after terminal close.

        Component-level disposal does not clear a shared Kernel unless explicitly
        requested. ``FormalTransportBundle.dispose`` performs the one-time
        shared-kernel cleanup for normal standalone/isolated-runner use.
        """

        if self._disposed:
            return self.lifecycle_diagnostics()
        self.close()
        self.completion_sink = None
        self.resource_release_sink = None
        self.authority_validation = None
        self.task_lookup = None
        self.permit_lookup = None
        self.resource_resolver = None
        self._reset_to_disposed_baseline()
        if dispose_kernel:
            self.kernel.dispose()
            self._mark_kernel_callbacks_disposed()
        self._disposed = True
        self._disposed_lifecycle_diagnostics = (
            self._build_lifecycle_diagnostics()
        )
        return self.lifecycle_diagnostics()

    def evidence(self) -> dict[str, Any]:
        return {
            "prepared_receipt_ids": tuple(sorted(self._prepared)),
            "confirmed_receipt_ids": tuple(sorted(self._confirmed_receipt_ids)),
            "aborted_receipt_ids": tuple(sorted(self._aborted_receipt_ids)),
            "receipt_states": tuple(
                sorted(
                    (*((receipt_id, "PREPARED") for receipt_id in self._prepared),
                     *((receipt_id, "CONFIRMED") for receipt_id in self._confirmed_receipt_ids),
                     *((receipt_id, "ABORTED") for receipt_id in self._aborted_receipt_ids))
                )
            ),
            "active_task_ids": tuple(sorted(self._active_transfers)),
            "started_task_ids": tuple(sorted(self._started_task_ids)),
            "completed_task_ids": tuple(sorted(self._completed_task_ids)),
            "busy_src_ranks": tuple(sorted(self._busy_src)),
            "busy_dst_ranks": tuple(sorted(self._busy_dst)),
            "busy_nic_ids": tuple(sorted(self._busy_nics)),
            "busy_lane_ids": tuple(sorted(self._busy_lanes)),
            "physical_record_digest": self.physical_record_digest(),
            "terminal_state": self.terminal_state(),
            "statistics": self.statistics(),
            "formal_runtime_metrics": self.formal_runtime_metrics(),
            "last_rejection": self._last_rejection,
            "internal_policy_wait_queue": False,
            "retained_rejected_batches": (),
        }

    def manifest_fragment(self) -> dict[str, Any]:
        return {
            "transport_transport": True,
            "transport_mode": "TRANSPORT_TRANSACTIONAL_DATA_PLANE",
            "hardware_profile_id": self.hardware_profile.profile_id,
            "hardware_profile_digest": self.hardware_profile.profile_digest,
            "profile_provenance": self.hardware_profile.profile_provenance,
            "performance_eligible": self.hardware_profile.performance_eligible,
            "topology_id": self.topology.topology_id,
            "topology_digest": self.topology.topology_digest,
            "resource_model": "RANK_TX_RX_NODE_NIC_AND_INDEPENDENT_LANE",
            "internal_policy_wait_queue": False,
            "evidence_schema": "TRANSPORT_DATA_PLANE_STATISTICS",
            "formal_runtime_metrics_schema": "TRANSPORT_DATA_RUNTIME_METRICS",
            "terminal_check_supported": True,
            "profile_sensitivity_input_supported": True,
            "phase_window_statistics_supported": True,
            "duplicate_event_fail_closed": True,
            "filtered_physical_metrics_supported": True,
            "physical_metrics_filter_semantics": "INTERSECTION_FAIL_CLOSED",
            "idempotent_terminal_close_supported": True,
            "idempotent_terminal_dispose_supported": True,
            "transport_owned_threads": 0,
            "transport_owned_executors": 0,
            "transport_owned_file_handles": 0,
            "transport_owned_child_processes": 0,
            **self.bandwidth_contention.manifest_fragment(),
        }

    def snapshot(self) -> TransportSnapshot:
        available = tuple(
            (
                link_class,
                tuple(lane for lane in lane_ids if lane not in self._busy_lanes),
            )
            for link_class, lane_ids in self.topology.lane_ids_by_link_class
        )
        return make_transport_snapshot(
            snapshot_at_ns=int(self.kernel.now_ns),
            max_batch_tasks=int(self.hardware_profile.max_batch_tasks),
            busy_src_ranks=tuple(self._busy_src),
            busy_dst_ranks=tuple(self._busy_dst),
            busy_nic_ids=tuple(self._busy_nics),
            busy_lane_ids=tuple(self._busy_lanes),
            available_lane_ids_by_link_class=available,
            hardware_profile_digest=self.hardware_profile.profile_digest,
            topology_digest=self.topology.topology_digest,
        )

    def _reject(
        self,
        outcome: SubmitOutcome,
        code: RejectionCode,
        detail: str,
        batch: TransferBatch | object,
        *,
        resource_kind: str | None = None,
    ) -> tuple[SubmitOutcome, None]:
        batch_id = str(getattr(batch, "batch_id", ""))
        self._last_rejection = TransportRejection(outcome, code, str(detail), batch_id)
        self._bump(self._rejection_counts, code.value)
        if code is RejectionCode.RESOURCE_BUSY:
            self._bump(
                self._resource_busy_retry_counts,
                resource_kind or "UNCLASSIFIED_RESOURCE",
            )
        phase_key = getattr(batch, "phase_key", None)
        if isinstance(phase_key, PhaseKey):
            self._record_mechanism_event(
                phase_key=phase_key,
                kind=f"REJECT_{code.value}",
                link_class=getattr(batch, "link_class", None),
                task_count=len(getattr(batch, "task_ids", ())),
            )
        return outcome, None

    @staticmethod
    def _task_matches_batch(task: CanonicalTransferTask, batch: TransferBatch) -> bool:
        return task.task_id in batch.task_ids and task.phase_key == batch.phase_key

    @staticmethod
    def _permit_matches_task(task: CanonicalTransferTask, permit: ReceivePermit | None) -> bool:
        return (
            permit is not None
            and permit.task_id == task.task_id
            and permit.edge_key == task.edge_key
            and permit.chunk_index == task.chunk_index
            and permit.byte_offset == task.byte_offset
            and permit.task_bytes == task.payload_bytes
            and permit.expectation_digest == task.expectation_digest
        )

    def _validate_footprint(
        self,
        task: CanonicalTransferTask,
        footprint: TaskResourceFootprint,
        batch: TransferBatch,
    ) -> bool:
        topology = self.topology
        if footprint.task_id != task.task_id:
            return False
        if footprint.topology_digest != topology.topology_digest:
            return False
        if task.src_rank < 0 or task.dst_rank < 0:
            return False
        if task.src_rank >= topology.world_size or task.dst_rank >= topology.world_size:
            return False
        expected_class = (
            LinkClass.INTRA_NODE
            if topology.rank_to_node[task.src_rank] == topology.rank_to_node[task.dst_rank]
            else LinkClass.INTER_NODE
        )
        if footprint.link_class != expected_class:
            return False
        if footprint.tx_nic_id != topology.tx_nic_id_by_rank[task.src_rank]:
            return False
        if footprint.rx_nic_id != topology.rx_nic_id_by_rank[task.dst_rank]:
            return False
        topology_lanes = set(dict(topology.lane_ids_by_link_class).get(expected_class, ()))
        eligible = set(footprint.eligible_lane_ids)
        return bool(eligible) and eligible.issubset(topology_lanes)

    def _contention_details(
        self, lane_assignment: dict[str, str]
    ) -> dict[str, tuple[str, int]]:
        """Return task -> (public contention group, deterministic share count)."""

        lane_to_group = dict(self.topology.nic_id_by_lane)
        if self.bandwidth_contention.mode == BANDWIDTH_MODE_FIXED_PER_LANE:
            return {
                task_id: (lane_id, 1)
                for task_id, lane_id in lane_assignment.items()
            }
        group_counts: dict[str, int] = {}
        group_by_task: dict[str, str] = {}
        for task_id, lane_id in lane_assignment.items():
            group_id = lane_to_group[lane_id]
            group_by_task[task_id] = group_id
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
        return {
            task_id: (group_id, group_counts[group_id])
            for task_id, group_id in group_by_task.items()
        }

    @staticmethod
    def _exact_lane_assignment(
        footprints: tuple[TaskResourceFootprint, ...],
        candidate_lanes: set[str],
    ) -> dict[str, str] | None:
        ordered = sorted(
            footprints,
            key=lambda value: (
                len(set(value.eligible_lane_ids) & candidate_lanes),
                value.task_id,
            ),
        )
        assignment: dict[str, str] = {}

        def visit(index: int, unused: set[str]) -> bool:
            if index == len(ordered):
                return True
            footprint = ordered[index]
            for lane_id in sorted(set(footprint.eligible_lane_ids) & unused):
                assignment[footprint.task_id] = lane_id
                if visit(index + 1, unused - {lane_id}):
                    return True
                assignment.pop(footprint.task_id, None)
            return False

        return assignment if visit(0, set(candidate_lanes)) else None

    def prepare_commit(
        self, batch: TransferBatch, commit_time_ns: int
    ) -> tuple[SubmitOutcome, CommitReceipt | None]:
        self._ensure_open()
        self._last_rejection = None
        self._prepare_attempt_count += 1
        if not isinstance(batch, TransferBatch):
            return self._reject(
                SubmitOutcome.FATAL_CONTRACT_ERROR,
                RejectionCode.INVALID_BATCH_TYPE,
                "batch must be TransferBatch",
                batch,
            )
        self._record_mechanism_event(
            phase_key=batch.phase_key,
            kind="PREPARE_ATTEMPT",
            link_class=batch.link_class,
            task_count=len(batch.task_ids),
            at_ns=int(self.kernel.now_ns),
        )
        if (
            not isinstance(commit_time_ns, int)
            or isinstance(commit_time_ns, bool)
            or commit_time_ns != int(self.kernel.now_ns)
        ):
            return self._reject(
                SubmitOutcome.FATAL_CONTRACT_ERROR,
                RejectionCode.INVALID_COMMIT_TIME,
                "commit_time_ns must equal kernel.now_ns",
                batch,
            )
        if len(batch.task_ids) > self.hardware_profile.max_batch_tasks:
            return self._reject(
                SubmitOutcome.FATAL_CONTRACT_ERROR,
                RejectionCode.BATCH_LIMIT_EXCEEDED,
                "batch exceeds hardware_profile.max_batch_tasks",
                batch,
            )
        if batch.topology_digest != self.topology.topology_digest:
            return self._reject(
                SubmitOutcome.FATAL_CONTRACT_ERROR,
                RejectionCode.TOPOLOGY_CONTRACT_MISMATCH,
                "batch topology digest differs from the injected topology",
                batch,
            )
        try:
            authority_current = self.authority_validation.authority_is_current(
                phase_key=batch.phase_key, authority_stamp=batch.authority_stamp
            )
        except Exception as exc:
            return self._reject(
                SubmitOutcome.FATAL_CONTRACT_ERROR,
                RejectionCode.TASK_CONTRACT_MISMATCH,
                f"authority validation port failed: {type(exc).__name__}: {exc}",
                batch,
            )
        if not authority_current:
            return self._reject(
                SubmitOutcome.RETRYABLE_STALE_AUTHORITY,
                RejectionCode.STALE_AUTHORITY,
                "authority stamp is not current",
                batch,
            )

        tasks: list[CanonicalTransferTask] = []
        permits: list[ReceivePermit] = []
        footprints: list[TaskResourceFootprint] = []
        for task_id in batch.task_ids:
            try:
                task = self.task_lookup.task(task_id)
            except Exception as exc:
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.TASK_LOOKUP_FAILED,
                    f"task lookup failed for {task_id}: {type(exc).__name__}: {exc}",
                    batch,
                )
            if not isinstance(task, CanonicalTransferTask) or not self._task_matches_batch(task, batch):
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.TASK_CONTRACT_MISMATCH,
                    f"task {task_id} does not match batch identity/phase",
                    batch,
                )
            if task.src_rank == task.dst_rank:
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.LOCAL_DIAGONAL_TASK,
                    f"local diagonal task {task_id} entered DataPlane",
                    batch,
                )
            if task.task_id in self._known_physical_task_ids or any(
                task.task_id in prepared.batch.task_ids for prepared in self._prepared.values()
            ):
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.DUPLICATE_PHYSICAL_TASK,
                    f"task {task_id} already has a physical reservation/record",
                    batch,
                )
            try:
                permit = self.permit_lookup.permit(task.task_id)
            except Exception as exc:
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.PERMIT_MISSING,
                    f"permit lookup failed for {task_id}: {type(exc).__name__}: {exc}",
                    batch,
                )
            if permit is None:
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.PERMIT_MISSING,
                    f"permit missing for {task_id}",
                    batch,
                )
            if not isinstance(permit, ReceivePermit) or not self._permit_matches_task(task, permit):
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.PERMIT_CONTRACT_MISMATCH,
                    f"permit does not bind task edge/chunk/range/bytes/expectation: {task_id}",
                    batch,
                )
            try:
                footprint = self.resource_resolver.footprint(task)
            except Exception as exc:
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.FOOTPRINT_LOOKUP_FAILED,
                    f"footprint lookup failed for {task_id}: {type(exc).__name__}: {exc}",
                    batch,
                )
            if not isinstance(footprint, TaskResourceFootprint) or not self._validate_footprint(
                task, footprint, batch
            ):
                code = (
                    RejectionCode.TOPOLOGY_CONTRACT_MISMATCH
                    if isinstance(footprint, TaskResourceFootprint)
                    and footprint.topology_digest != self.topology.topology_digest
                    else RejectionCode.FOOTPRINT_CONTRACT_MISMATCH
                )
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    code,
                    f"resource footprint does not match task/batch/topology: {task_id}",
                    batch,
                )
            tasks.append(task)
            permits.append(permit)
            footprints.append(footprint)

        if {footprint.link_class for footprint in footprints} != {batch.link_class}:
            return self._reject(
                SubmitOutcome.FATAL_CONTRACT_ERROR,
                RejectionCode.MIXED_LINK_CLASS,
                "batch contains mixed or mismatched link classes",
                batch,
            )
        srcs = [task.src_rank for task in tasks]
        dsts = [task.dst_rank for task in tasks]
        if len(set(srcs)) != len(srcs) or len(set(dsts)) != len(dsts):
            return self._reject(
                SubmitOutcome.FATAL_CONTRACT_ERROR,
                RejectionCode.INTERNAL_ENDPOINT_CONFLICT,
                "batch contains conflicting rank TX or RX endpoints",
                batch,
            )
        batch_nics: set[str] = set()
        for footprint in footprints:
            task_nics = {footprint.tx_nic_id, footprint.rx_nic_id}
            if task_nics & batch_nics:
                return self._reject(
                    SubmitOutcome.FATAL_CONTRACT_ERROR,
                    RejectionCode.INTERNAL_NIC_CONFLICT,
                    "batch contains conflicting node NIC resources",
                    batch,
                )
            batch_nics.update(task_nics)

        all_lanes = set(dict(self.topology.lane_ids_by_link_class).get(batch.link_class, ()))
        theoretical_assignment = self._exact_lane_assignment(tuple(footprints), all_lanes)
        if theoretical_assignment is None:
            return self._reject(
                SubmitOutcome.FATAL_CONTRACT_ERROR,
                RejectionCode.NO_LEGAL_LANE_ASSIGNMENT,
                "batch has no legal lane assignment even on an idle topology",
                batch,
            )

        if set(srcs) & self._busy_src or set(dsts) & self._busy_dst:
            return self._reject(
                SubmitOutcome.RETRYABLE_RESOURCE_BUSY,
                RejectionCode.RESOURCE_BUSY,
                "rank TX/RX resource is busy",
                batch,
                resource_kind="RANK_ENDPOINT",
            )
        if batch_nics & self._busy_nics:
            return self._reject(
                SubmitOutcome.RETRYABLE_RESOURCE_BUSY,
                RejectionCode.RESOURCE_BUSY,
                "node NIC resource is busy",
                batch,
                resource_kind="NIC",
            )
        lane_assignment = self._exact_lane_assignment(
            tuple(footprints), all_lanes - self._busy_lanes
        )
        if lane_assignment is None:
            return self._reject(
                SubmitOutcome.RETRYABLE_RESOURCE_BUSY,
                RejectionCode.RESOURCE_BUSY,
                "eligible lane resource is busy",
                batch,
                resource_kind="LANE",
            )

        launch_delay = self._map(
            self.hardware_profile.launch_delay_ns_by_link_class
        )[batch.link_class]
        fixed_latency = self._map(
            self.hardware_profile.fixed_latency_ns_by_link_class
        )[batch.link_class]
        bandwidth = self._map(
            self.hardware_profile.bandwidth_bytes_per_second_by_link_class
        )[batch.link_class]
        start_at_ns = int(commit_time_ns) + int(launch_delay)
        contention_details = self._contention_details(lane_assignment)
        reserved: list[_ProvisionalTaskReservation] = []
        contention_rows: list[tuple[str, str, int, int, int]] = []
        for task, permit, footprint in zip(tasks, permits, footprints):
            contention_group_id, contention_share_count = contention_details[
                task.task_id
            ]
            base_transfer_time_ns = ceil_transfer_time_ns(
                task.payload_bytes, int(bandwidth)
            )
            effective_transfer_time_ns = ceil_transfer_time_ns(
                task.payload_bytes * int(contention_share_count), int(bandwidth)
            )
            complete_at_ns = (
                start_at_ns
                + int(fixed_latency)
                + int(effective_transfer_time_ns)
            )
            reserved.append(
                _ProvisionalTaskReservation(
                    task=task,
                    permit=permit,
                    footprint=footprint,
                    lane_id=lane_assignment[task.task_id],
                    committed_at_ns=int(commit_time_ns),
                    start_at_ns=start_at_ns,
                    complete_at_ns=complete_at_ns,
                )
            )
            contention_rows.append(
                (
                    task.task_id,
                    contention_group_id,
                    int(contention_share_count),
                    int(base_transfer_time_ns),
                    int(effective_transfer_time_ns),
                )
            )

        reservation_digest = stable_digest(
            tuple(reserved), domain="TRANSPORT_RESOURCE_RESERVATION"
        )
        snapshot_digest = stable_digest(self.snapshot(), domain="TRANSPORT_SNAPSHOT")
        receipt = make_commit_receipt(
            batch=batch,
            commit_time_ns=int(commit_time_ns),
            resource_reservation_digest=reservation_digest,
            transport_snapshot_digest=snapshot_digest,
        )

        # This is the sole mutation point in prepare and occurs only after all
        # validation and complete atomic resource assignment have succeeded.
        for item in reserved:
            self._busy_src.add(item.task.src_rank)
            self._busy_dst.add(item.task.dst_rank)
            self._busy_nics.update(
                {item.footprint.tx_nic_id, item.footprint.rx_nic_id}
            )
            self._busy_lanes.add(item.lane_id)
        self._prepared[receipt.receipt_id] = _PreparedCommit(
            receipt=receipt,
            batch=batch,
            reserved=tuple(reserved),
            contention_by_task=tuple(sorted(contention_rows)),
            prepared_at_kernel_ns=int(self.kernel.now_ns),
        )
        self._prepared_commit_count_total += 1
        self._record_mechanism_event(
            phase_key=batch.phase_key,
            kind="PREPARED",
            link_class=batch.link_class,
            task_count=len(batch.task_ids),
            payload_bytes=sum(task.payload_bytes for task in tasks),
            at_ns=int(self.kernel.now_ns),
        )
        return SubmitOutcome.PREPARED, receipt

    def confirm_commit(self, receipt: CommitReceipt) -> None:
        self._ensure_open()
        prepared = self._prepared.get(getattr(receipt, "receipt_id", ""))
        if prepared is None or prepared.receipt != receipt:
            raise ReceiptStateError("unknown, changed, aborted, or already confirmed receipt")
        if int(self.kernel.now_ns) != prepared.prepared_at_kernel_ns:
            raise ReceiptStateError("prepared receipt is no longer live at the current timestamp")

        # From this point the exact live receipt is valid.  No fallible external
        # validation remains; event identities are unique because physical task
        # identity is globally single-use in this transport.
        self._prepared.pop(receipt.receipt_id)
        self._confirmed_receipt_ids.add(receipt.receipt_id)
        self._confirmed_receipt_task_ids[receipt.receipt_id] = tuple(
            prepared.batch.task_ids
        )
        self._confirmed_commit_count += 1
        link_class = prepared.batch.link_class
        self._bump(self._launch_count_by_link_class, link_class)
        launch_cost_ns = self._map(
            self.hardware_profile.launch_delay_ns_by_link_class
        )[link_class]
        self._bump(
            self._launch_cost_ns_by_link_class, link_class, int(launch_cost_ns)
        )
        committed_bytes = sum(item.task.payload_bytes for item in prepared.reserved)
        self._bump(self._committed_bytes_by_link_class, link_class, committed_bytes)
        self._record_mechanism_event(
            phase_key=prepared.batch.phase_key,
            kind="CONFIRMED",
            link_class=link_class,
            task_count=len(prepared.reserved),
            payload_bytes=committed_bytes,
            duration_ns=int(launch_cost_ns),
            at_ns=int(self.kernel.now_ns),
        )
        if prepared.reserved:
            common_start = min(item.start_at_ns for item in prepared.reserved)
            batch_complete = max(item.complete_at_ns for item in prepared.reserved)
            self._bump(
                self._batch_span_busy_time_ns_by_link_class,
                link_class,
                batch_complete - common_start,
            )
            self._first_physical_start_ns = (
                common_start
                if self._first_physical_start_ns is None
                else min(self._first_physical_start_ns, common_start)
            )
            self._last_physical_completion_ns = (
                batch_complete
                if self._last_physical_completion_ns is None
                else max(self._last_physical_completion_ns, batch_complete)
            )
        contention_by_task = {
            task_id: (group_id, share_count, base_ns, effective_ns)
            for task_id, group_id, share_count, base_ns, effective_ns
            in prepared.contention_by_task
        }
        for item in prepared.reserved:
            transfer_duration_ns = item.complete_at_ns - item.start_at_ns
            resource_busy_duration_ns = item.complete_at_ns - item.committed_at_ns
            self._bump(
                self._transfer_busy_time_ns_by_link_class,
                link_class,
                transfer_duration_ns,
            )
            self._bump(
                self._lane_busy_time_ns, item.lane_id, resource_busy_duration_ns
            )
            for nic_id in {item.footprint.tx_nic_id, item.footprint.rx_nic_id}:
                self._bump(
                    self._nic_busy_time_ns, nic_id, resource_busy_duration_ns
                )
            self._bump(
                self._rank_tx_busy_time_ns,
                item.task.src_rank,
                resource_busy_duration_ns,
            )
            self._bump(
                self._rank_rx_busy_time_ns,
                item.task.dst_rank,
                resource_busy_duration_ns,
            )
            self._physical_task_by_id[item.task.task_id] = item.task
            self._resource_footprint_by_task[item.task.task_id] = item.footprint
            (
                contention_group_id,
                contention_share_count,
                base_transfer_time_ns,
                effective_transfer_time_ns,
            ) = contention_by_task[item.task.task_id]
            self._contention_detail_by_task[item.task.task_id] = (
                contention_group_id,
                contention_share_count,
                base_transfer_time_ns,
                effective_transfer_time_ns,
            )
            if contention_share_count > 1:
                self._contention_affected_task_count += 1
                self._contention_extra_service_ns += (
                    effective_transfer_time_ns - base_transfer_time_ns
                )
            self._max_contention_share_count = max(
                self._max_contention_share_count, contention_share_count
            )
            record = PhysicalTransferRecord(
                task_id=item.task.task_id,
                batch_id=prepared.batch.batch_id,
                link_class=prepared.batch.link_class,
                lane_id=item.lane_id,
                committed_at_ns=item.committed_at_ns,
                start_at_ns=item.start_at_ns,
                complete_at_ns=item.complete_at_ns,
                payload_bytes=item.task.payload_bytes,
            )
            self._known_physical_task_ids.add(record.task_id)
            self._active_transfers[record.task_id] = _ActiveTransfer(
                task=item.task, footprint=item.footprint, record=record
            )
            self._records_by_task[record.task_id] = record
            event = self.kernel.schedule(
                time_ns=record.start_at_ns,
                phase_priority=KernelPhase.COMPLETION_COLLECTION,
                producer=self.PRODUCER,
                event_type=self.START_EVENT,
                ordinal=semantic_ordinal(
                    "start", receipt.receipt_id, record.task_id, record.start_at_ns
                ),
                subject_id=record.task_id,
            )
            self._event_record_by_id[event.stable_event_id] = record

    def abort_commit(self, receipt: CommitReceipt) -> None:
        self._ensure_open()
        receipt_id = str(getattr(receipt, "receipt_id", ""))
        if receipt_id in self._aborted_receipt_ids:
            return
        if receipt_id in self._confirmed_receipt_ids:
            raise ReceiptStateError("confirmed receipt cannot be aborted")
        prepared = self._prepared.get(receipt_id)
        if prepared is None or prepared.receipt != receipt:
            raise ReceiptStateError("unknown or changed receipt")
        if int(self.kernel.now_ns) != prepared.prepared_at_kernel_ns:
            raise ReceiptStateError("prepared receipt is no longer live at the current timestamp")
        self._prepared.pop(receipt_id)
        self._release_reserved(prepared.reserved)
        self._aborted_receipt_ids.add(receipt_id)
        self._aborted_commit_count += 1
        self._record_mechanism_event(
            phase_key=prepared.batch.phase_key,
            kind="ABORTED",
            link_class=prepared.batch.link_class,
            task_count=len(prepared.reserved),
            payload_bytes=sum(item.task.payload_bytes for item in prepared.reserved),
            at_ns=int(self.kernel.now_ns),
        )

    def _release_reserved(
        self, reserved: tuple[_ProvisionalTaskReservation, ...]
    ) -> None:
        for item in reserved:
            self._busy_src.discard(item.task.src_rank)
            self._busy_dst.discard(item.task.dst_rank)
            self._busy_nics.discard(item.footprint.tx_nic_id)
            self._busy_nics.discard(item.footprint.rx_nic_id)
            self._busy_lanes.discard(item.lane_id)

    def _handle_start(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        record = self._event_record_by_id.pop(event.stable_event_id, None)
        if record is None:
            raise ReceiptStateError(
                f"duplicate or unknown transfer start event: {event.stable_event_id}"
            )
        if record.task_id in self._started_task_ids:
            raise ReceiptStateError(f"duplicate transfer start for task {record.task_id}")
        if record.task_id not in self._active_transfers:
            raise ReceiptStateError(f"start event references inactive task {record.task_id}")
        self._started_task_ids.add(record.task_id)
        record_digest = stable_digest(record, domain="PHYSICAL_TRANSFER_RECORD")
        self.completion_sink.on_transfer_started(
            TransferStarted(
                task_id=record.task_id,
                batch_id=record.batch_id,
                start_at_ns=record.start_at_ns,
                physical_record_digest=record_digest,
            )
        )
        completion = kernel.schedule(
            time_ns=record.complete_at_ns,
            phase_priority=KernelPhase.COMPLETION_COLLECTION,
            producer=self.PRODUCER,
            event_type=self.COMPLETE_EVENT,
            ordinal=semantic_ordinal(
                "complete", record.batch_id, record.task_id, record.complete_at_ns
            ),
            subject_id=record.task_id,
        )
        self._event_record_by_id[completion.stable_event_id] = record
        self._record_mechanism_event(
            phase_key=self._active_transfers[record.task_id].task.phase_key,
            kind="STARTED",
            link_class=record.link_class,
            task_count=1,
            payload_bytes=record.payload_bytes,
            at_ns=record.start_at_ns,
        )
        return ProgressSignal(authoritative_state_updates=1, notes=("transport_start",))

    def _handle_complete(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        del kernel
        record = self._event_record_by_id.pop(event.stable_event_id, None)
        if record is None:
            raise ReceiptStateError(
                f"duplicate or unknown transfer completion event: {event.stable_event_id}"
            )
        if record.task_id in self._completed_task_ids:
            raise ReceiptStateError(f"duplicate transfer completion for task {record.task_id}")
        active = self._active_transfers.get(record.task_id)
        if active is None:
            raise ReceiptStateError(
                f"completion event references inactive task {record.task_id}"
            )
        if record.task_id not in self._started_task_ids:
            raise ReceiptStateError(
                f"completion event precedes start for task {record.task_id}"
            )
        record_digest = stable_digest(record, domain="PHYSICAL_TRANSFER_RECORD")
        self.completion_sink.on_transfer_completed(
            TransferCompleted(
                task_id=record.task_id,
                batch_id=record.batch_id,
                complete_at_ns=record.complete_at_ns,
                payload_bytes=record.payload_bytes,
                physical_record_digest=record_digest,
            )
        )
        task = active.task
        footprint = active.footprint
        self._bump(
            self._completed_bytes_by_link_class,
            record.link_class,
            record.payload_bytes,
        )
        self._record_mechanism_event(
            phase_key=task.phase_key,
            kind="COMPLETED",
            link_class=record.link_class,
            task_count=1,
            payload_bytes=record.payload_bytes,
            duration_ns=record.complete_at_ns - record.start_at_ns,
            at_ns=record.complete_at_ns,
        )
        self._active_transfers.pop(record.task_id, None)
        self._completed_task_ids.add(record.task_id)
        self._busy_src.discard(task.src_rank)
        self._busy_dst.discard(task.dst_rank)
        self._busy_nics.discard(footprint.tx_nic_id)
        self._busy_nics.discard(footprint.rx_nic_id)
        self._busy_lanes.discard(record.lane_id)
        self.resource_release_sink.on_transport_resources_released(task.phase_key)
        return ProgressSignal(
            authoritative_state_updates=1,
            notes=("transport_complete_and_release",),
        )


__all__ = ["FormalDataPlaneTransport"]
