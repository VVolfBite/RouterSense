from __future__ import annotations

"""Formal backend ↔ transport ControlPlane adapter and backend runtime-facing factory.

The factory in this module assembles backend and receiver state around
an injected kernel bridge, observer and formal ControlPlane.  It deliberately
does not construct the simulation kernel, Scheduler, DataPlane, topology or a
second complete runtime; those remain the Integration Owner's responsibility.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

from rs_sim import (
    ControlPlaneDelivery,
    ExactRowDescriptor,
    make_row_broadcast_request,
    stable_digest,
)
from rs_sim.contracts.paper_defaults import (
    PAPER_P0_P1_COMPUTE_END_BARRIER, PAPER_RELEASE_MODE,
)
from rs_sim.backend import (
    AttributeSharedObjectAdapter,
    BackendTraceFixtureBuilder,
    LinearReceiverCostModel,
    ReceiverService,
    SimulationBackend,
)
from rs_sim.transport.api.ports import ControlPlaneDeliverySink, ControlPlaneTransportPort

from ..assembly.bindings import (
    SchemaEdgeKeyFactory,
    SchemaExpectationFactory,
    SchemaPermitFactory,
    make_phase_semantics,
)


@dataclass(slots=True)
class BackendControlPlaneAdapter(ControlPlaneDeliverySink):
    """One-way exact-row publication and delayed expectation materialization.

    Publication creates only a public ``RowBroadcastRequest``. Backend-owned
    Dispatch expectations are created later, in ``on_control_plane_delivery``.
    """

    backend: Any
    control_plane: ControlPlaneTransportPort
    _descriptor_by_request_digest: dict[str, ExactRowDescriptor] = field(
        default_factory=dict
    )
    _delivered_request_digests: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.control_plane.attach_delivery_sink(self)

    def publish_exact_descriptor(self, descriptor: ExactRowDescriptor) -> str:
        if not isinstance(descriptor, ExactRowDescriptor):
            raise TypeError("formal ControlPlane adapter requires ExactRowDescriptor")
        request = make_row_broadcast_request(descriptor)
        expected_digest = stable_digest(request, domain="ROW_BROADCAST_REQUEST")

        if expected_digest in self._delivered_request_digests:
            # Exact replay after delivery is idempotent and must not schedule a
            # second ControlPlane event.
            return expected_digest
        existing = self._descriptor_by_request_digest.get(expected_digest)
        if existing is not None:
            if existing != descriptor:
                raise RuntimeError("request digest collision across exact descriptors")
            return expected_digest

        # Register the correlation before invoking the ControlPlane. A conforming transport schedules
        # delivery at the Kernel descriptor-delivery phase, but pre-registration
        # also makes the adapter safe against a protocol fake that calls back
        # synchronously.
        self._descriptor_by_request_digest[expected_digest] = descriptor
        try:
            request_digest = self.control_plane.publish_row(request)
        except Exception:
            self._descriptor_by_request_digest.pop(expected_digest, None)
            raise
        if request_digest != expected_digest:
            self._descriptor_by_request_digest.pop(expected_digest, None)
            raise RuntimeError("ControlPlane returned a non-canonical request digest")
        return request_digest

    def on_control_plane_delivery(self, delivery: ControlPlaneDelivery) -> None:
        if not isinstance(delivery, ControlPlaneDelivery):
            raise TypeError("delivery must be ControlPlaneDelivery")
        if delivery.request_digest in self._delivered_request_digests:
            raise RuntimeError("duplicate ControlPlane delivery")
        try:
            descriptor = self._descriptor_by_request_digest[delivery.request_digest]
        except KeyError as exc:
            raise RuntimeError("unknown ControlPlane delivery request digest") from exc
        if delivery.phase_key != descriptor.phase_key or delivery.src_rank != descriptor.src_rank:
            raise RuntimeError("ControlPlane delivery does not match exact descriptor")
        if delivery.delivery_start_ns < descriptor.published_at_ns:
            raise RuntimeError("ControlPlane delivery started before descriptor publication")

        # This is the only point at which descriptor-derived Dispatch
        # expectations may become visible to backend/scheduler.
        self.backend.on_exact_row_descriptor_delivered(
            descriptor=descriptor,
            delivered_at_ns=delivery.delivered_at_ns,
        )
        self._descriptor_by_request_digest.pop(delivery.request_digest)
        self._delivered_request_digests.add(delivery.request_digest)

    @property
    def pending_request_digests(self) -> tuple[str, ...]:
        return tuple(sorted(self._descriptor_by_request_digest))

    @property
    def delivered_request_digests(self) -> tuple[str, ...]:
        return tuple(sorted(self._delivered_request_digests))


@dataclass(slots=True)
class BackendRuntimeDriver:
    """Runtime-facing backend bundle with no Scheduler or DataPlane ownership."""

    backend: SimulationBackend
    receiver: ReceiverService
    trace_builder: BackendTraceFixtureBuilder
    control_adapter: BackendControlPlaneAdapter

    def register_fixture(self, **kwargs: Any) -> Any:
        return self.trace_builder.register_fixture(**kwargs)

    def terminal_state(self, registration: Any) -> dict[str, Any]:
        phase_states = tuple(
            self.backend.phase_terminal_snapshot(phase_key=phase_key)
            for phase_key in registration.all_phase_keys
        )
        memory = {
            rank: self.receiver.current_memory(rank)
            for rank in range(self.backend.world_size)
        }
        rank_states = {
            rank: self.backend.rank_state(rank).value
            for rank in range(self.backend.world_size)
        }
        receiver_memory_zero = all(
            int(values["total_receiver_bytes"]) == 0
            for values in memory.values()
        )
        ranks_done = all(state == "DONE" for state in rank_states.values())
        terminal = bool(
            phase_states
            and all(state["closed"] for state in phase_states)
            and receiver_memory_zero
            and ranks_done
            and not self.control_adapter.pending_request_digests
        )
        return {
            "terminal": terminal,
            "phase_states": phase_states,
            "receiver_memory_by_rank": memory,
            "rank_states": rank_states,
            "pending_control_request_digests": (
                self.control_adapter.pending_request_digests
            ),
        }

    def assert_terminal(self, registration: Any) -> dict[str, Any]:
        state = self.terminal_state(registration)
        if not state["terminal"]:
            raise RuntimeError(f"Backend runtime is not terminal: {state}")
        return state


def build_backend_runtime_driver(
    *,
    world_size: int,
    staging_capacity_bytes_by_rank: Mapping[int, int | None],
    kernel_bridge: Any,
    observer: Any,
    control_plane: ControlPlaneTransportPort,
    release_mode: str = PAPER_RELEASE_MODE,
    p0_p1_compute_end_barrier: bool = PAPER_P0_P1_COMPUTE_END_BARRIER,
    node_id_by_rank: Mapping[int, int] | None = None,
    receiver_cost_model: Any | None = None,
    local_assembly_cost_ns: int = 0,
) -> BackendRuntimeDriver:
    """Build only the production backend slice around injected public dependencies."""

    adapter = AttributeSharedObjectAdapter()
    semantics = make_phase_semantics()
    receiver = ReceiverService(
        world_size=int(world_size),
        staging_capacity_bytes_by_rank=dict(staging_capacity_bytes_by_rank),
        kernel=kernel_bridge,
        observer=observer,
        adapter=adapter,
        phase_semantics=semantics,
        permit_factory=SchemaPermitFactory(),
        cost_model=receiver_cost_model
        or LinearReceiverCostModel(
            posting_fixed_ns=1,
            posting_bytes_per_ns=10**9,
            drain_fixed_ns=1,
            drain_bytes_per_ns=10**9,
        ),
        local_assembly_cost_ns=int(local_assembly_cost_ns),
    )
    backend = SimulationBackend(
        world_size=int(world_size),
        kernel=kernel_bridge,
        observer=observer,
        adapter=adapter,
        phase_semantics=semantics,
        edge_key_factory=SchemaEdgeKeyFactory(),
        expectation_factory=SchemaExpectationFactory(),
        receiver=receiver,
        release_mode=release_mode,
        p0_p1_compute_end_barrier=p0_p1_compute_end_barrier,
        node_id_by_rank=node_id_by_rank,
    )
    attach_backend = getattr(kernel_bridge, "attach_backend", None)
    if attach_backend is not None:
        attach_backend(backend)
    control_adapter = BackendControlPlaneAdapter(
        backend=backend,
        control_plane=control_plane,
    )
    backend.attach_exact_row_publisher(control_adapter)
    return BackendRuntimeDriver(
        backend=backend,
        receiver=receiver,
        trace_builder=BackendTraceFixtureBuilder(backend=backend),
        control_adapter=control_adapter,
    )


__all__ = [
    "BackendControlPlaneAdapter",
    "BackendRuntimeDriver",
    "build_backend_runtime_driver",
]
