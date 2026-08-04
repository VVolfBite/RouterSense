from __future__ import annotations

"""Scheduler-side ports for transport integration.

This module exposes adapters/builders only.  It does not own or assemble the
integrated runtime; Wave B remains the Integration Owner's responsibility.
"""

from dataclasses import dataclass
from typing import Any

from rs_sim import (
    AuthorityStamp,
    CanonicalTransferTask,
    PhaseKey,
    TransferCompleted,
    TransferStarted,
)
from rs_sim.scheduler.execution.compiler import FormalTransportResourceAdapter
from rs_sim.transport.api.ports import (
    AuthorityValidationPort,
    CompletionSink,
    ResourceReleaseSink,
    TaskLookupPort,
    TaskResourceResolverPort,
)


@dataclass(frozen=True, slots=True)
class SchedulerTaskLookupAdapter(TaskLookupPort):
    catalogue: Any

    def task(self, task_id: str) -> CanonicalTransferTask:
        task = self.catalogue.get(str(task_id))
        if not isinstance(task, CanonicalTransferTask):
            raise TypeError("formal TaskLookupPort requires CanonicalTransferTask")
        return task


@dataclass(frozen=True, slots=True)
class SchedulerAuthorityValidationAdapter(AuthorityValidationPort):
    authority: Any

    def authority_is_current(
        self, *, phase_key: PhaseKey, authority_stamp: AuthorityStamp
    ) -> bool:
        if not isinstance(phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        if not isinstance(authority_stamp, AuthorityStamp):
            return False
        return bool(self.authority.stamp_is_current(phase_key, authority_stamp))


@dataclass(slots=True)
class SchedulerBackendCompletionAdapter(CompletionSink):
    """Apply the frozen start/completion order across scheduler and backend.

    Start: transport -> scheduler mark_running.
    Completion: transport -> scheduler mark_completed -> backend receiver completion.
    Physical resources are released by transport only after this callback returns.
    """

    authority: Any
    catalogue: Any
    backend: Any
    completion_tracker: Any | None = None

    def _task(self, task_id: str) -> CanonicalTransferTask:
        task = self.catalogue.get(str(task_id))
        if not isinstance(task, CanonicalTransferTask):
            raise TypeError("canonical task must be CanonicalTransferTask")
        return task

    def on_transfer_started(self, event: TransferStarted) -> None:
        if not isinstance(event, TransferStarted):
            raise TypeError("event must be TransferStarted")
        task = self._task(event.task_id)
        self.authority.mark_running(
            task.phase_key,
            event.task_id,
            at_ns=event.start_at_ns,
        )

    def on_transfer_completed(self, event: TransferCompleted) -> None:
        if not isinstance(event, TransferCompleted):
            raise TypeError("event must be TransferCompleted")
        task = self._task(event.task_id)
        if int(event.payload_bytes) != int(task.payload_bytes):
            raise ValueError("TransferCompleted payload_bytes does not match canonical task")
        self.authority.mark_completed(
            task.phase_key,
            event.task_id,
            at_ns=event.complete_at_ns,
        )
        self.backend.on_transfer_completed(
            task_id=event.task_id,
            at_ns=event.complete_at_ns,
        )
        if self.completion_tracker is not None:
            self.completion_tracker.on_task_completed(
                task.phase_key, at_ns=event.complete_at_ns
            )


@dataclass(slots=True)
class SchedulerResourceReleaseAdapter(ResourceReleaseSink):
    """Forward resource release only to execution stabilization."""

    scheduler_bridge: Any

    def on_transport_resources_released(self, phase_key: PhaseKey) -> None:
        if not isinstance(phase_key, PhaseKey):
            raise TypeError("phase_key must be PhaseKey")
        self.scheduler_bridge.notify_transport_resource_release(phase_key)


@dataclass(frozen=True, slots=True)
class SchedulerPortBundle:
    """scheduler-owned public ports consumed by transport and the common compiler."""

    task_lookup: SchedulerTaskLookupAdapter
    authority_validation: SchedulerAuthorityValidationAdapter
    completion_sink: SchedulerBackendCompletionAdapter
    resource_release_sink: SchedulerResourceReleaseAdapter
    resource_adapter: FormalTransportResourceAdapter


def build_scheduler_port_bundle(
    *,
    catalogue: Any,
    authority: Any,
    backend: Any,
    scheduler_bridge: Any,
    resource_resolver: TaskResourceResolverPort,
    expected_hardware_profile_digest: str,
    completion_tracker: Any | None = None,
) -> SchedulerPortBundle:
    """Build formal scheduler adapters around one shared topology resolver."""

    task_lookup = SchedulerTaskLookupAdapter(catalogue=catalogue)
    authority_validation = SchedulerAuthorityValidationAdapter(authority=authority)
    completion_sink = SchedulerBackendCompletionAdapter(
        authority=authority,
        catalogue=catalogue,
        backend=backend,
        completion_tracker=completion_tracker,
    )
    resource_release_sink = SchedulerResourceReleaseAdapter(
        scheduler_bridge=scheduler_bridge
    )
    resource_adapter = FormalTransportResourceAdapter(
        task_lookup=task_lookup,
        resource_resolver=resource_resolver,
        expected_hardware_profile_digest=str(expected_hardware_profile_digest),
    )
    return SchedulerPortBundle(
        task_lookup=task_lookup,
        authority_validation=authority_validation,
        completion_sink=completion_sink,
        resource_release_sink=resource_release_sink,
        resource_adapter=resource_adapter,
    )


__all__ = [
    "SchedulerAuthorityValidationAdapter",
    "SchedulerBackendCompletionAdapter",
    "SchedulerPortBundle",
    "SchedulerResourceReleaseAdapter",
    "SchedulerTaskLookupAdapter",
    "build_scheduler_port_bundle",
]
