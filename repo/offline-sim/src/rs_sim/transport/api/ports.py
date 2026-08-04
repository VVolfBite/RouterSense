from __future__ import annotations

from typing import Protocol

from rs_sim.contracts.schema import (
    AuthorityStamp,
    CanonicalTransferTask,
    CommitReceipt,
    ControlPlaneDelivery,
    NetworkTopology,
    PhaseKey,
    ReceivePermit,
    RowBroadcastRequest,
    SubmitOutcome,
    TaskResourceFootprint,
    TransferBatch,
    TransferCompleted,
    TransferStarted,
    TransportSnapshot,
)


class TaskLookupPort(Protocol):
    def task(self, task_id: str) -> CanonicalTransferTask: ...


class PermitLookupPort(Protocol):
    def permit(self, task_id: str) -> ReceivePermit | None: ...


class AuthorityValidationPort(Protocol):
    def authority_is_current(
        self, *, phase_key: PhaseKey, authority_stamp: AuthorityStamp
    ) -> bool: ...


class TaskResourceResolverPort(Protocol):
    @property
    def topology(self) -> NetworkTopology: ...

    def footprint(self, task: CanonicalTransferTask) -> TaskResourceFootprint: ...


class CompletionSink(Protocol):
    def on_transfer_started(self, event: TransferStarted) -> None: ...
    def on_transfer_completed(self, event: TransferCompleted) -> None: ...


class ResourceReleaseSink(Protocol):
    def on_transport_resources_released(self, phase_key: PhaseKey) -> None: ...


class ControlPlaneDeliverySink(Protocol):
    def on_control_plane_delivery(self, delivery: ControlPlaneDelivery) -> None: ...


class DataPlaneTransportPort(Protocol):
    def snapshot(self) -> TransportSnapshot: ...
    def prepare_commit(
        self, batch: TransferBatch, commit_time_ns: int
    ) -> tuple[SubmitOutcome, CommitReceipt | None]: ...
    def confirm_commit(self, receipt: CommitReceipt) -> None:
        """Infallible for an unconsumed, live receipt returned by prepare_commit."""
        ...
    def abort_commit(self, receipt: CommitReceipt) -> None: ...


class ControlPlaneTransportPort(Protocol):
    def attach_delivery_sink(self, sink: ControlPlaneDeliverySink) -> None: ...
    def publish_row(self, request: RowBroadcastRequest) -> str: ...


__all__ = [
    "AuthorityValidationPort",
    "CompletionSink",
    "ControlPlaneDeliverySink",
    "ControlPlaneTransportPort",
    "DataPlaneTransportPort",
    "PermitLookupPort",
    "ResourceReleaseSink",
    "TaskLookupPort",
    "TaskResourceResolverPort",
]
