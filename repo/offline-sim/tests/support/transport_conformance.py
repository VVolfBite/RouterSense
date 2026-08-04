from __future__ import annotations

"""Protocol-level transport conformance fake for contract tests only.

It exercises public transport ports and immutable schema without importing backend
or scheduler private state.  It is deterministic but not performance-eligible and is
not the formal transport implementation.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

from rs_sim import (
    CommitReceipt,
    ControlPlaneDelivery,
    ControlPlaneProfile,
    HardwareProfile,
    KernelPhase,
    LinkClass,
    PhysicalTransferRecord,
    ProgressSignal,
    RowBroadcastRequest,
    SimulationEvent,
    SimulationKernel,
    SubmitOutcome,
    TransferBatch,
    TransferCompleted,
    TransferStarted,
    TransportSnapshot,
    ceil_transfer_time_ns,
    make_commit_receipt,
    make_transport_snapshot,
    stable_digest,
    stable_json_dumps,
)
from rs_sim.transport.api.ports import (
    AuthorityValidationPort,
    CompletionSink,
    ControlPlaneDeliverySink,
    PermitLookupPort,
    ResourceReleaseSink,
    TaskLookupPort,
    TaskResourceResolverPort,
)


def _ordinal(*parts: Any) -> int:
    text = stable_json_dumps(tuple(parts))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


@dataclass(frozen=True, slots=True)
class _Prepared:
    receipt: CommitReceipt
    batch: TransferBatch
    records: tuple[PhysicalTransferRecord, ...]


class TransportConformanceFake:
    """Public-port-only transport used to prove transport/backend/scheduler interface closure."""

    START_EVENT = "TRANSPORT_CONFORMANCE_TRANSFER_START"
    COMPLETE_EVENT = "TRANSPORT_CONFORMANCE_TRANSFER_COMPLETE"
    CONTROL_DELIVERY_EVENT = "TRANSPORT_CONFORMANCE_CONTROL_DELIVERY"

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
        control_profile: ControlPlaneProfile,
    ) -> None:
        if hardware_profile.performance_eligible or control_profile.performance_eligible:
            raise ValueError("conformance fake requires non-performance synthetic profiles")
        self.kernel = kernel
        self.task_lookup = task_lookup
        self.permit_lookup = permit_lookup
        self.authority_validation = authority_validation
        self.resource_resolver = resource_resolver
        self.completion_sink = completion_sink
        self.resource_release_sink = resource_release_sink
        self.hardware_profile = hardware_profile
        self.control_profile = control_profile
        self._busy_src: set[int] = set()
        self._busy_dst: set[int] = set()
        self._busy_nics: set[str] = set()
        self._busy_lanes: set[str] = set()
        self._prepared: dict[str, _Prepared] = {}
        self._committed: dict[str, PhysicalTransferRecord] = {}
        self._event_payload: dict[str, Any] = {}
        self._control_sink: ControlPlaneDeliverySink | None = None
        self._control_available_at_ns = 0
        self._control_delivery_by_request: dict[str, ControlPlaneDelivery] = {}
        kernel.register_event_handler(self.START_EVENT, self._handle_start)
        kernel.register_event_handler(self.COMPLETE_EVENT, self._handle_complete)
        kernel.register_event_handler(self.CONTROL_DELIVERY_EVENT, self._handle_control_delivery)

    @property
    def topology(self):
        return self.resource_resolver.topology

    @staticmethod
    def _profile_map(rows):
        return dict(rows)

    def snapshot(self) -> TransportSnapshot:
        available = []
        for link_class, lane_ids in self.topology.lane_ids_by_link_class:
            available.append(
                (link_class, tuple(lane for lane in lane_ids if lane not in self._busy_lanes))
            )
        return make_transport_snapshot(
            snapshot_at_ns=int(self.kernel.now_ns),
            max_batch_tasks=int(self.hardware_profile.max_batch_tasks),
            busy_src_ranks=tuple(sorted(self._busy_src)),
            busy_dst_ranks=tuple(sorted(self._busy_dst)),
            busy_nic_ids=tuple(sorted(self._busy_nics)),
            busy_lane_ids=tuple(sorted(self._busy_lanes)),
            available_lane_ids_by_link_class=tuple(available),
            hardware_profile_digest=self.hardware_profile.profile_digest,
            topology_digest=self.topology.topology_digest,
        )

    def _validate_task_permit(self, task, permit) -> bool:
        return (
            permit is not None
            and permit.task_id == task.task_id
            and permit.edge_key == task.edge_key
            and permit.chunk_index == task.chunk_index
            and permit.byte_offset == task.byte_offset
            and permit.task_bytes == task.payload_bytes
            and permit.expectation_digest == task.expectation_digest
        )

    def _reserve_records(
        self, batch: TransferBatch, commit_time_ns: int
    ) -> tuple[PhysicalTransferRecord, ...] | None:
        tasks = tuple(self.task_lookup.task(task_id) for task_id in batch.task_ids)
        if any(task.src_rank == task.dst_rank for task in tasks):
            return None
        footprints = tuple(self.resource_resolver.footprint(task) for task in tasks)
        if any(fp.topology_digest != batch.topology_digest for fp in footprints):
            return None
        if {fp.link_class for fp in footprints} != {batch.link_class}:
            return None
        srcs = [task.src_rank for task in tasks]
        dsts = [task.dst_rank for task in tasks]
        if len(set(srcs)) != len(srcs) or len(set(dsts)) != len(dsts):
            return None
        if set(srcs) & self._busy_src or set(dsts) & self._busy_dst:
            return ()
        nics: set[str] = set()
        lanes_available = set(dict(self.snapshot().available_lane_ids_by_link_class).get(batch.link_class, ()))
        assigned: list[str] = []
        for fp in sorted(footprints, key=lambda value: value.task_id):
            task_nics = {fp.tx_nic_id, fp.rx_nic_id}
            if task_nics & self._busy_nics or task_nics & nics:
                return ()
            nics.update(task_nics)
            choices = sorted(set(fp.eligible_lane_ids) & lanes_available)
            if not choices:
                return ()
            lane = choices[0]
            lanes_available.remove(lane)
            assigned.append(lane)
        launch_delay = self._profile_map(
            self.hardware_profile.launch_delay_ns_by_link_class
        )[batch.link_class]
        fixed_latency = self._profile_map(
            self.hardware_profile.fixed_latency_ns_by_link_class
        )[batch.link_class]
        bandwidth = self._profile_map(
            self.hardware_profile.bandwidth_bytes_per_second_by_link_class
        )[batch.link_class]
        start_at = int(commit_time_ns) + int(launch_delay)
        records = []
        for task, lane in zip(tasks, assigned):
            complete_at = start_at + int(fixed_latency) + ceil_transfer_time_ns(
                task.payload_bytes, int(bandwidth)
            )
            records.append(
                PhysicalTransferRecord(
                    task_id=task.task_id,
                    batch_id=batch.batch_id,
                    link_class=batch.link_class,
                    lane_id=lane,
                    committed_at_ns=int(commit_time_ns),
                    start_at_ns=start_at,
                    complete_at_ns=complete_at,
                    payload_bytes=task.payload_bytes,
                )
            )
        return tuple(records)

    def prepare_commit(
        self, batch: TransferBatch, commit_time_ns: int
    ) -> tuple[SubmitOutcome, CommitReceipt | None]:
        if not isinstance(batch, TransferBatch):
            return SubmitOutcome.FATAL_CONTRACT_ERROR, None
        if batch.topology_digest != self.topology.topology_digest:
            return SubmitOutcome.FATAL_CONTRACT_ERROR, None
        if not self.authority_validation.authority_is_current(
            phase_key=batch.phase_key, authority_stamp=batch.authority_stamp
        ):
            return SubmitOutcome.RETRYABLE_STALE_AUTHORITY, None
        tasks = tuple(self.task_lookup.task(task_id) for task_id in batch.task_ids)
        if any(
            not self._validate_task_permit(task, self.permit_lookup.permit(task.task_id))
            for task in tasks
        ):
            return SubmitOutcome.FATAL_CONTRACT_ERROR, None
        records = self._reserve_records(batch, int(commit_time_ns))
        if records is None:
            return SubmitOutcome.FATAL_CONTRACT_ERROR, None
        if records == ():
            return SubmitOutcome.RETRYABLE_RESOURCE_BUSY, None
        reservation_digest = stable_digest(records, domain="TRANSPORT_CONFORMANCE_RESERVATION")
        snapshot_digest = stable_digest(self.snapshot(), domain="TRANSPORT_SNAPSHOT")
        receipt = make_commit_receipt(
            batch=batch,
            commit_time_ns=int(commit_time_ns),
            resource_reservation_digest=reservation_digest,
            transport_snapshot_digest=snapshot_digest,
        )
        for task, record in zip(tasks, records):
            fp = self.resource_resolver.footprint(task)
            self._busy_src.add(task.src_rank)
            self._busy_dst.add(task.dst_rank)
            self._busy_nics.update({fp.tx_nic_id, fp.rx_nic_id})
            self._busy_lanes.add(record.lane_id)
        self._prepared[receipt.receipt_id] = _Prepared(receipt, batch, records)
        return SubmitOutcome.PREPARED, receipt

    def abort_commit(self, receipt: CommitReceipt) -> None:
        prepared = self._prepared.pop(receipt.receipt_id)
        if prepared.receipt != receipt:
            raise RuntimeError("CommitReceipt mutated before abort")
        self._release_records(prepared.records)

    def confirm_commit(self, receipt: CommitReceipt) -> None:
        """Infallible for a valid, live receipt returned by prepare_commit."""
        prepared = self._prepared.pop(receipt.receipt_id)
        # No validation remains here by contract.  All records are already
        # immutable and resources are reserved.
        for record in prepared.records:
            self._committed[record.task_id] = record
            event = self.kernel.schedule(
                time_ns=record.start_at_ns,
                phase_priority=KernelPhase.COMPLETION_COLLECTION,
                producer="TRANSPORT_CONFORMANCE",
                event_type=self.START_EVENT,
                ordinal=_ordinal("start", record.task_id, record.start_at_ns),
                subject_id=record.task_id,
            )
            self._event_payload[event.stable_event_id] = record

    def _handle_start(self, kernel: SimulationKernel, event: SimulationEvent) -> ProgressSignal:
        record = self._event_payload.pop(event.stable_event_id)
        digest = stable_digest(record, domain="PHYSICAL_TRANSFER_RECORD")
        self.completion_sink.on_transfer_started(
            TransferStarted(
                task_id=record.task_id,
                batch_id=record.batch_id,
                start_at_ns=record.start_at_ns,
                physical_record_digest=digest,
            )
        )
        completion = kernel.schedule(
            time_ns=record.complete_at_ns,
            phase_priority=KernelPhase.COMPLETION_COLLECTION,
            producer="TRANSPORT_CONFORMANCE",
            event_type=self.COMPLETE_EVENT,
            ordinal=_ordinal("complete", record.task_id, record.complete_at_ns),
            subject_id=record.task_id,
        )
        self._event_payload[completion.stable_event_id] = record
        return ProgressSignal(authoritative_state_updates=1, notes=("transport_start",))

    def _handle_complete(self, kernel: SimulationKernel, event: SimulationEvent) -> ProgressSignal:
        del kernel
        record = self._event_payload.pop(event.stable_event_id)
        digest = stable_digest(record, domain="PHYSICAL_TRANSFER_RECORD")
        self.completion_sink.on_transfer_completed(
            TransferCompleted(
                task_id=record.task_id,
                batch_id=record.batch_id,
                complete_at_ns=record.complete_at_ns,
                payload_bytes=record.payload_bytes,
                physical_record_digest=digest,
            )
        )
        self._committed.pop(record.task_id)
        self._release_records((record,))
        task = self.task_lookup.task(record.task_id)
        self.resource_release_sink.on_transport_resources_released(task.phase_key)
        return ProgressSignal(authoritative_state_updates=1, notes=("transport_complete",))

    def _release_records(self, records: tuple[PhysicalTransferRecord, ...]) -> None:
        for record in records:
            task = self.task_lookup.task(record.task_id)
            fp = self.resource_resolver.footprint(task)
            self._busy_src.discard(task.src_rank)
            self._busy_dst.discard(task.dst_rank)
            self._busy_nics.discard(fp.tx_nic_id)
            self._busy_nics.discard(fp.rx_nic_id)
            self._busy_lanes.discard(record.lane_id)

    def attach_delivery_sink(self, sink: ControlPlaneDeliverySink) -> None:
        if self._control_sink is not None and self._control_sink is not sink:
            raise RuntimeError("ControlPlane delivery sink already attached")
        self._control_sink = sink

    def publish_row(self, request: RowBroadcastRequest) -> str:
        if self._control_sink is None:
            raise RuntimeError("ControlPlane delivery sink is not attached")
        request_digest = stable_digest(request, domain="ROW_BROADCAST_REQUEST")
        start_at = max(int(request.published_at_ns), self._control_available_at_ns)
        duration = int(self.control_profile.fixed_latency_ns) + ceil_transfer_time_ns(
            int(request.descriptor_payload_bytes),
            int(self.control_profile.bandwidth_bytes_per_second),
        )
        delivered_at = start_at + duration
        self._control_available_at_ns = delivered_at
        delivery = ControlPlaneDelivery(
            request_digest=request_digest,
            phase_key=request.phase_key,
            src_rank=request.src_rank,
            delivery_start_ns=start_at,
            delivered_at_ns=delivered_at,
            control_channel_id="control-fifo-0",
        )
        self._control_delivery_by_request[request_digest] = delivery
        event = self.kernel.schedule(
            time_ns=delivered_at,
            phase_priority=KernelPhase.DESCRIPTOR_OBSERVATION_DELIVERY,
            producer="TRANSPORT_CONFORMANCE_CONTROL",
            event_type=self.CONTROL_DELIVERY_EVENT,
            ordinal=_ordinal("control", request_digest, delivered_at),
            subject_id=request_digest,
        )
        self._event_payload[event.stable_event_id] = delivery
        return request_digest

    def _handle_control_delivery(
        self, kernel: SimulationKernel, event: SimulationEvent
    ) -> ProgressSignal:
        del kernel
        delivery = self._event_payload.pop(event.stable_event_id)
        assert self._control_sink is not None
        self._control_sink.on_control_plane_delivery(delivery)
        return ProgressSignal(authoritative_state_updates=1, notes=("control_delivery",))

    @property
    def prepared_count(self) -> int:
        return len(self._prepared)

    @property
    def committed_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._committed))


__all__ = ["TransportConformanceFake"]
