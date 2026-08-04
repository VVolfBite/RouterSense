from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rs_sim.backend.core.simulation import SimulationKernel
from rs_sim.contracts.schema import (
    CommitReceipt,
    ControlPlaneProfile,
    HardwareProfile,
    RowBroadcastRequest,
    SubmitOutcome,
    TransferBatch,
    TransportSnapshot,
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

from .builders import FormalTransportBundle, build_formal_transports
from ..core.errors import TransportRejection
from ..config.profiles import (
    BandwidthContentionSensitivity,
    TransportProfileBundle,
    TransportProfileProvider,
)
from ..observability.metrics import PhysicalMetricsView


@dataclass(frozen=True, slots=True)
class AtomicBatchSubmission:
    outcome: SubmitOutcome
    receipt: CommitReceipt | None
    applied: bool
    confirmed: bool
    rejection: TransportRejection | None


class FormalTransportRuntimeDriver:
    """Public Integration-Owner facade over the formal transport transaction.

    The driver contains no policy queue and does not advance the kernel. It only
    wires the atomic prepare/apply/confirm protocol and aborts a live receipt if
    the Scheduler's logical receipt application raises.
    """

    def __init__(self, bundle: FormalTransportBundle) -> None:
        if not isinstance(bundle, FormalTransportBundle):
            raise TypeError("bundle must be FormalTransportBundle")
        self.bundle = bundle

    @property
    def data_plane(self):
        return self.bundle.data_plane

    @property
    def control_plane(self):
        return self.bundle.control_plane

    def snapshot(self) -> TransportSnapshot:
        return self.data_plane.snapshot()

    def publish_dispatch_row(self, request: RowBroadcastRequest) -> str:
        return self.control_plane.publish_row(request)

    def submit_atomic_batch(
        self,
        *,
        batch: TransferBatch,
        commit_time_ns: int,
        apply_receipt: Callable[[CommitReceipt], Any],
    ) -> AtomicBatchSubmission:
        if not callable(apply_receipt):
            raise TypeError("apply_receipt must be callable")
        outcome, receipt = self.data_plane.prepare_commit(batch, commit_time_ns)
        if outcome is not SubmitOutcome.PREPARED:
            return AtomicBatchSubmission(
                outcome=outcome,
                receipt=None,
                applied=False,
                confirmed=False,
                rejection=self.data_plane.last_rejection,
            )
        assert receipt is not None
        try:
            apply_receipt(receipt)
        except BaseException:
            self.data_plane.abort_commit(receipt)
            raise
        self.data_plane.confirm_commit(receipt)
        return AtomicBatchSubmission(
            outcome=outcome,
            receipt=receipt,
            applied=True,
            confirmed=True,
            rejection=None,
        )

    def abort_prepared(self, receipt: CommitReceipt) -> None:
        self.data_plane.abort_commit(receipt)

    def manifest_fragment(self) -> dict[str, Any]:
        return {
            **self.bundle.manifest_fragment(),
            "transport_runtime_driver": "TRANSPORT_RUNTIME_DRIVER",
            "transport_driver_internal_policy_queue": False,
            "transport_driver_advances_kernel": False,
        }

    def statistics(self) -> dict[str, Any]:
        return self.bundle.statistics()

    def formal_runtime_metrics(self) -> dict[str, Any]:
        return self.bundle.formal_runtime_metrics()

    def physical_metrics(
        self,
        *,
        phase_keys=(),
        task_ids=(),
        window_task_ids=(),
    ) -> PhysicalMetricsView:
        return self.data_plane.physical_metrics(
            phase_keys=phase_keys,
            task_ids=task_ids,
            window_task_ids=window_task_ids,
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "schema_version": "TRANSPORT_RUNTIME_DRIVER_EVIDENCE",
            "manifest_fragment": self.manifest_fragment(),
            "bundle": self.bundle.evidence(),
            "formal_runtime_metrics": self.formal_runtime_metrics(),
        }

    def terminal_state(self) -> dict[str, Any]:
        return self.bundle.terminal_state()

    def assert_terminal(self) -> None:
        self.bundle.assert_terminal()

    def lifecycle_diagnostics(self) -> dict[str, Any]:
        return self.bundle.lifecycle_diagnostics()

    def close(self) -> dict[str, Any]:
        return self.bundle.close()

    def dispose(self, *, dispose_kernel: bool = True) -> dict[str, Any]:
        return self.bundle.dispose(dispose_kernel=dispose_kernel)


def build_formal_transport_runtime_driver(
    *,
    kernel: SimulationKernel,
    task_lookup: TaskLookupPort,
    permit_lookup: PermitLookupPort,
    authority_validation: AuthorityValidationPort,
    resource_resolver: TaskResourceResolverPort,
    completion_sink: CompletionSink,
    resource_release_sink: ResourceReleaseSink,
    control_delivery_sink: ControlPlaneDeliverySink | None = None,
    hardware_profile: HardwareProfile | None = None,
    control_profile: ControlPlaneProfile | None = None,
    profile_bundle: TransportProfileBundle | None = None,
    profile_provider: TransportProfileProvider | None = None,
    bandwidth_contention: BandwidthContentionSensitivity | None = None,
    formal_execution_mode: str = "ORDER_ONLY",
) -> FormalTransportRuntimeDriver:
    return FormalTransportRuntimeDriver(
        build_formal_transports(
            kernel=kernel,
            task_lookup=task_lookup,
            permit_lookup=permit_lookup,
            authority_validation=authority_validation,
            resource_resolver=resource_resolver,
            completion_sink=completion_sink,
            resource_release_sink=resource_release_sink,
            control_delivery_sink=control_delivery_sink,
            hardware_profile=hardware_profile,
            control_profile=control_profile,
            profile_bundle=profile_bundle,
            profile_provider=profile_provider,
            bandwidth_contention=bandwidth_contention,
            formal_execution_mode=formal_execution_mode,
        )
    )


__all__ = [
    "AtomicBatchSubmission",
    "FormalTransportRuntimeDriver",
    "build_formal_transport_runtime_driver",
]
