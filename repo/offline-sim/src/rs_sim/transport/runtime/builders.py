from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs_sim.contracts.factories import make_control_plane_profile, make_hardware_profile
from rs_sim.backend.core.simulation import SimulationKernel
from rs_sim.contracts.schema import ControlPlaneProfile, HardwareProfile, LinkClass
from rs_sim.contracts.digest import stable_digest
from rs_sim.transport.api.ports import (
    AuthorityValidationPort,
    CompletionSink,
    ControlPlaneDeliverySink,
    PermitLookupPort,
    ResourceReleaseSink,
    TaskLookupPort,
    TaskResourceResolverPort,
)

from ..control.channel import FormalControlPlaneTransport
from ..data.channel import FormalDataPlaneTransport
from ..core.errors import validate_formal_execution_mode
from ..config.profiles import (
    BandwidthContentionSensitivity,
    TransportProfileBundle,
    TransportProfileProvider,
    fixed_per_lane_bandwidth_sensitivity,
)


def make_default_synthetic_hardware_profile(
    *, max_batch_tasks: int
) -> HardwareProfile:
    """Correctness-only profile; values are not hardware calibration claims."""

    return make_hardware_profile(
        profile_id="rs-sim-synthetic-data-v1",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        max_batch_tasks=int(max_batch_tasks),
        launch_delay_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 100),
            (LinkClass.INTER_NODE, 1_000),
        ),
        fixed_latency_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 1_000),
            (LinkClass.INTER_NODE, 5_000),
        ),
        bandwidth_bytes_per_second_by_link_class=(
            (LinkClass.INTRA_NODE, 100_000_000_000),
            (LinkClass.INTER_NODE, 25_000_000_000),
        ),
    )


def make_default_synthetic_control_profile() -> ControlPlaneProfile:
    """Correctness-only independent ControlPlane profile."""

    return make_control_plane_profile(
        profile_id="rs-sim-synthetic-control-v1",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        fixed_latency_ns=1_000,
        bandwidth_bytes_per_second=10_000_000_000,
    )


@dataclass(frozen=True, slots=True)
class FormalTransportBundle:
    data_plane: FormalDataPlaneTransport
    control_plane: FormalControlPlaneTransport
    profile_bundle: TransportProfileBundle | None = None
    formal_execution_mode: str = "ORDER_ONLY"

    def manifest_fragment(self) -> dict[str, Any]:
        data = self.data_plane.manifest_fragment()
        control = self.control_plane.manifest_fragment()
        profile_fragment = (
            self.profile_bundle.manifest_fragment()
            if self.profile_bundle is not None
            else {
                "transport_profile_handoff_mode": "DIRECT_COMPONENT_PROFILES",
                "transport_profile_bundle_id": None,
                "transport_profile_bundle_digest": None,
            }
        )
        return {
            **data,
            **control,
            **profile_fragment,
            "transport_evidence_version": "TRANSPORT_BUNDLE_EVIDENCE",
            "transport_terminal_check_supported": True,
            "transport_runtime_metrics_supported": True,
            "transport_performance_eligible": bool(
                data["performance_eligible"]
                and control["control_plane_performance_eligible"]
            ),
            "transport_execution_mode": self.formal_execution_mode,
            "formal_full_joint_status": "EXPERIMENTAL_BLOCKED_NOT_LIVE",
        }

    def evidence(self) -> dict[str, Any]:
        return {
            "schema_version": "TRANSPORT_BUNDLE_EVIDENCE",
            "manifest_fragment": self.manifest_fragment(),
            "data_plane": self.data_plane.evidence(),
            "control_plane": self.control_plane.evidence(),
            "formal_runtime_metrics": self.formal_runtime_metrics(),
            "terminal_state": self.terminal_state(),
        }

    def statistics(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "TRANSPORT_BUNDLE_STATISTICS",
            "data_plane": self.data_plane.statistics(),
            "control_plane": self.control_plane.statistics(),
        }
        payload["statistics_digest"] = stable_digest(
            payload, domain="TRANSPORT_BUNDLE_STATISTICS"
        )
        return payload

    def formal_runtime_metrics(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "TRANSPORT_RUNTIME_METRICS",
            "manifest_fragment": self.manifest_fragment(),
            "data_plane": self.data_plane.formal_runtime_metrics(),
            "control_plane": self.control_plane.formal_runtime_metrics(),
            "terminal_state": self.terminal_state(),
        }
        payload["runtime_metrics_digest"] = stable_digest(
            payload, domain="TRANSPORT_RUNTIME_METRICS"
        )
        return payload

    def terminal_state(self) -> dict[str, Any]:
        data = self.data_plane.terminal_state()
        control = self.control_plane.terminal_state()
        return {
            "terminal": bool(data["terminal"] and control["terminal"]),
            "data_plane": data,
            "control_plane": control,
        }

    def assert_statistics_reconcile(self) -> None:
        self.data_plane.assert_statistics_reconcile()
        self.control_plane.assert_statistics_reconcile()

    def assert_terminal(self) -> None:
        state = self.terminal_state()
        if not state["terminal"]:
            raise RuntimeError(f"Formal transport bundle is not terminal: {state}")
        self.assert_statistics_reconcile()

    def lifecycle_diagnostics(self) -> dict[str, Any]:
        data = self.data_plane.lifecycle_diagnostics()
        control = self.control_plane.lifecycle_diagnostics()
        return {
            "schema_version": "TRANSPORT_BUNDLE_LIFECYCLE_DIAGNOSTICS",
            "formal_execution_mode": self.formal_execution_mode,
            "kernel_pending_event_count": int(
                self.data_plane.kernel.pending_event_count()
            ),
            "data_plane": data,
            "control_plane": control,
            "closed": bool(data["closed"] and control["closed"]),
            "disposed": bool(data["disposed"] and control["disposed"]),
            "kernel_callback_registry_disposed": bool(
                data["kernel_callback_registry_disposed"]
                and control["kernel_callback_registry_disposed"]
            ),
        }

    def close(self) -> dict[str, Any]:
        self.data_plane.close()
        self.control_plane.close()
        return self.lifecycle_diagnostics()

    def dispose(self, *, dispose_kernel: bool = True) -> dict[str, Any]:
        kernel = self.data_plane.kernel
        if kernel is not self.control_plane.kernel:
            raise RuntimeError("formal transport bundle components do not share one Kernel")
        self.data_plane.dispose(dispose_kernel=False)
        self.control_plane.dispose(dispose_kernel=False)
        if dispose_kernel:
            kernel.dispose()
            self.data_plane._mark_kernel_callbacks_disposed()
            self.control_plane._mark_kernel_callbacks_disposed()
        return self.lifecycle_diagnostics()


def _resolve_profiles(
    *,
    world_size: int,
    hardware_profile: HardwareProfile | None,
    control_profile: ControlPlaneProfile | None,
    profile_bundle: TransportProfileBundle | None,
    profile_provider: TransportProfileProvider | None,
    bandwidth_contention: BandwidthContentionSensitivity | None,
) -> tuple[
    HardwareProfile,
    ControlPlaneProfile,
    BandwidthContentionSensitivity,
    TransportProfileBundle | None,
]:
    if profile_bundle is not None and profile_provider is not None:
        raise ValueError("provide at most one of profile_bundle and profile_provider")
    if profile_provider is not None:
        loaded = profile_provider.load_transport_profile_bundle()
        if not isinstance(loaded, TransportProfileBundle):
            raise TypeError("profile provider returned a non-TransportProfileBundle value")
        profile_bundle = loaded
    if profile_bundle is not None:
        if hardware_profile is not None or control_profile is not None:
            raise ValueError(
                "component profiles cannot be combined with profile_bundle/profile_provider"
            )
        if (
            bandwidth_contention is not None
            and bandwidth_contention != profile_bundle.bandwidth_contention
        ):
            raise ValueError(
                "explicit bandwidth_contention conflicts with profile bundle"
            )
        return (
            profile_bundle.hardware_profile,
            profile_bundle.control_profile,
            profile_bundle.bandwidth_contention,
            profile_bundle,
        )
    data_profile = hardware_profile or make_default_synthetic_hardware_profile(
        max_batch_tasks=max(1, int(world_size))
    )
    cp_profile = control_profile or make_default_synthetic_control_profile()
    contention = bandwidth_contention or fixed_per_lane_bandwidth_sensitivity()
    return data_profile, cp_profile, contention, None


def build_formal_transports(
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
) -> FormalTransportBundle:
    normalized_execution_mode = validate_formal_execution_mode(
        formal_execution_mode
    )
    topology = resource_resolver.topology
    data_profile, cp_profile, contention, resolved_bundle = _resolve_profiles(
        world_size=topology.world_size,
        hardware_profile=hardware_profile,
        control_profile=control_profile,
        profile_bundle=profile_bundle,
        profile_provider=profile_provider,
        bandwidth_contention=bandwidth_contention,
    )
    data_plane = FormalDataPlaneTransport(
        kernel=kernel,
        task_lookup=task_lookup,
        permit_lookup=permit_lookup,
        authority_validation=authority_validation,
        resource_resolver=resource_resolver,
        completion_sink=completion_sink,
        resource_release_sink=resource_release_sink,
        hardware_profile=data_profile,
        bandwidth_contention=contention,
    )
    control_plane = FormalControlPlaneTransport(
        kernel=kernel,
        profile=cp_profile,
        delivery_sink=control_delivery_sink,
    )
    return FormalTransportBundle(
        data_plane=data_plane,
        control_plane=control_plane,
        profile_bundle=resolved_bundle,
        formal_execution_mode=normalized_execution_mode,
    )


__all__ = [
    "FormalTransportBundle",
    "build_formal_transports",
    "make_default_synthetic_control_profile",
    "make_default_synthetic_hardware_profile",
]
