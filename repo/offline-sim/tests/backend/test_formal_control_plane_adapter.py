from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from rs_sim import (
    CanonicalTransferTask,
    ControlPlaneDelivery,
    EdgeKey,
    PhaseKey,
    PhaseKind,
    TransferCompleted,
    stable_digest,
)
from rs_sim.backend import (
    AttributeSharedObjectAdapter,
    IllegalTransitionError,
    LinearReceiverCostModel,
    ReceiverService,
    SimulationBackend,
)
from rs_sim.runtime.adapters.backend import BackendControlPlaneAdapter
from rs_sim.runtime.assembly.bindings import (
    SchemaEdgeKeyFactory,
    SchemaExpectationFactory,
    SchemaPermitFactory,
    make_phase_semantics,
)
from rs_sim.runtime.adapters.kernel import BackendKernelBridge
from rs_sim.runtime.adapters.scheduler import SchedulerBackendCompletionAdapter
from rs_sim.backend.core.simulation import SimulationKernel
from rs_sim.contracts.schema import RowBroadcastRequest
from rs_sim.transport.api.ports import ControlPlaneDeliverySink


class Observer:
    def __init__(self) -> None:
        self.rows: list[tuple[str, int, dict[str, Any]]] = []

    def emit(self, *, kind: str, at_ns: int, payload: Mapping[str, Any]) -> None:
        self.rows.append((kind, at_ns, dict(payload)))


class DeferredControlPlaneFake:
    """Protocol-level ControlPlane fake; it never owns Backend state or time."""

    def __init__(self) -> None:
        self.sink: ControlPlaneDeliverySink | None = None
        self.requests: dict[str, RowBroadcastRequest] = {}
        self.publish_calls: list[str] = []

    def attach_delivery_sink(self, sink: ControlPlaneDeliverySink) -> None:
        if self.sink is not None and self.sink is not sink:
            raise RuntimeError("delivery sink already attached")
        self.sink = sink

    def publish_row(self, request: RowBroadcastRequest) -> str:
        request_digest = stable_digest(request, domain="ROW_BROADCAST_REQUEST")
        self.requests[request_digest] = request
        self.publish_calls.append(request_digest)
        return request_digest

    def deliver(
        self,
        request_digest: str,
        *,
        delivery_start_ns: int,
        delivered_at_ns: int,
    ) -> None:
        assert self.sink is not None
        request = self.requests[request_digest]
        self.sink.on_control_plane_delivery(
            ControlPlaneDelivery(
                request_digest=request_digest,
                phase_key=request.phase_key,
                src_rank=request.src_rank,
                delivery_start_ns=delivery_start_ns,
                delivered_at_ns=delivered_at_ns,
                control_channel_id="backend-control-test-0",
            )
        )


@dataclass
class FormalSystem:
    kernel: SimulationKernel
    backend: SimulationBackend
    receiver: ReceiverService
    observer: Observer
    control: DeferredControlPlaneFake
    control_adapter: BackendControlPlaneAdapter


def build_formal_system(
    *,
    world_size: int = 2,
    capacity_bytes: int = 16,
    local_assembly_cost_ns: int = 0,
) -> FormalSystem:
    kernel = SimulationKernel()
    kernel_bridge = BackendKernelBridge(kernel)
    observer = Observer()
    object_adapter = AttributeSharedObjectAdapter()
    phase_semantics = make_phase_semantics()
    receiver = ReceiverService(
        world_size=world_size,
        staging_capacity_bytes_by_rank={
            rank: capacity_bytes for rank in range(world_size)
        },
        kernel=kernel_bridge,
        observer=observer,
        adapter=object_adapter,
        phase_semantics=phase_semantics,
        permit_factory=SchemaPermitFactory(),
        cost_model=LinearReceiverCostModel(
            posting_fixed_ns=1,
            posting_bytes_per_ns=10**9,
            drain_fixed_ns=1,
            drain_bytes_per_ns=10**9,
        ),
        local_assembly_cost_ns=local_assembly_cost_ns,
    )
    backend = SimulationBackend(
        world_size=world_size,
        kernel=kernel_bridge,
        observer=observer,
        adapter=object_adapter,
        phase_semantics=phase_semantics,
        edge_key_factory=SchemaEdgeKeyFactory(),
        expectation_factory=SchemaExpectationFactory(),
        receiver=receiver,
        release_mode="RANK_LOCAL",
    )
    kernel_bridge.attach_backend(backend)
    control = DeferredControlPlaneFake()
    control_adapter = BackendControlPlaneAdapter(
        backend=backend,
        control_plane=control,
    )
    backend.attach_exact_row_publisher(control_adapter)
    return FormalSystem(kernel, backend, receiver, observer, control, control_adapter)


def drain_kernel(kernel: SimulationKernel) -> None:
    while kernel.has_events():
        kernel.run_next_timestamp()


def run_kernel_through(kernel: SimulationKernel, limit_ns: int) -> None:
    while kernel.has_events() and (kernel.next_event_time_ns() or 0) <= limit_ns:
        kernel.run_next_timestamp()


def register_truth(
    system: FormalSystem,
    *,
    phase: PhaseKey,
    src_rank: int,
    rows: tuple[int, ...],
    dispatch_bytes: tuple[int, ...],
    combine_bytes: tuple[int, ...],
) -> None:
    system.backend.register_exact_dispatch_row_truth(
        phase_key=phase,
        src_rank=src_rank,
        realized_rows_by_destination=rows,
        dispatch_payload_bytes_by_destination=dispatch_bytes,
        combine_return_payload_bytes_by_expert=combine_bytes,
        dispatch_payload_spec_digest="dispatch-spec-v1",
        combine_payload_spec_digest="combine-spec-v1",
        descriptor_payload_bytes=24,
    )


def test_descriptor_digest_is_content_only_and_delivery_gates_expectations() -> None:
    phase = PhaseKey("run", "sample", 0, PhaseKind.DISPATCH)
    results = []
    for published_at_ns in (5, 15):
        system = build_formal_system()
        register_truth(
            system,
            phase=phase,
            src_rank=0,
            rows=(1, 2),
            dispatch_bytes=(4, 8),
            combine_bytes=(6, 12),
        )
        system.backend.on_bootstrap_dispatch_local_path_complete(
            phase_key=phase,
            rank_id=0,
            at_ns=published_at_ns,
        )
        drain_kernel(system.kernel)

        descriptor = system.backend.exact_row_descriptor(
            phase_key=phase, src_rank=0
        )
        request_digest = system.backend.control_request_digest(
            phase_key=phase, src_rank=0
        )
        assert descriptor is not None
        assert request_digest is not None
        assert system.receiver.edges_by_key == {}
        assert system.control_adapter.pending_request_digests == (request_digest,)

        system.control.deliver(
            request_digest,
            delivery_start_ns=published_at_ns + 1,
            delivered_at_ns=published_at_ns + 3,
        )
        expectations = system.receiver.expected_edges_for_destination(
            phase_key=phase, dst_rank=1
        )
        assert len(expectations) == 1
        results.append(
            (
                descriptor.descriptor_digest,
                request_digest,
                expectations[0].expectation_digest,
            )
        )

    assert results[0][0] == results[1][0]
    assert results[0][2] == results[1][2]
    # The ControlPlane request includes publication time even though workload
    # identity and downstream task identity do not.
    assert results[0][1] != results[1][1]


def test_adapter_publication_is_idempotent_and_rejects_late_backend_attachment() -> None:
    phase = PhaseKey("run", "sample", 0, PhaseKind.DISPATCH)
    system = build_formal_system()
    register_truth(
        system,
        phase=phase,
        src_rank=0,
        rows=(1, 0),
        dispatch_bytes=(4, 0),
        combine_bytes=(6, 0),
    )
    system.backend.on_bootstrap_dispatch_local_path_complete(
        phase_key=phase, rank_id=0, at_ns=0
    )
    descriptor = system.backend.exact_row_descriptor(phase_key=phase, src_rank=0)
    assert descriptor is not None
    first = system.control_adapter.publish_exact_descriptor(descriptor)
    second = system.control_adapter.publish_exact_descriptor(descriptor)
    assert first == second
    assert system.control.publish_calls == [first]

    with pytest.raises(IllegalTransitionError):
        system.backend.attach_exact_row_publisher(system.control_adapter)


def test_zero_edges_close_dispatch_without_tasks_or_permits() -> None:
    phase = PhaseKey("run", "sample", 0, PhaseKind.DISPATCH)
    system = build_formal_system()
    register_truth(
        system,
        phase=phase,
        src_rank=0,
        rows=(0, 2),
        dispatch_bytes=(0, 8),
        combine_bytes=(0, 12),
    )
    register_truth(
        system,
        phase=phase,
        src_rank=1,
        rows=(0, 0),
        dispatch_bytes=(0, 0),
        combine_bytes=(0, 0),
    )
    for rank in range(2):
        system.backend.on_bootstrap_dispatch_local_path_complete(
            phase_key=phase, rank_id=rank, at_ns=0
        )
    drain_kernel(system.kernel)

    request_by_src = {
        request.src_rank: digest for digest, request in system.control.requests.items()
    }
    system.control.deliver(
        request_by_src[0], delivery_start_ns=1, delivered_at_ns=2
    )
    system.control.deliver(
        request_by_src[1], delivery_start_ns=2, delivered_at_ns=3
    )

    snapshot = system.backend.dispatch_destination_snapshot(
        phase_key=phase, dst_rank=0
    )
    edges = system.receiver.expected_edges_for_destination(
        phase_key=phase, dst_rank=0
    )
    assert snapshot["descriptor_closure_at_ns"] == 3
    assert len(edges) == 2 and all(edge.zero_edge for edge in edges)
    assert system.receiver.tasks_by_key == {}
    assert all(edge.task_ids == [] for edge in edges)
    complete, complete_at = system.receiver.all_nonzero_inbound_assembled(
        phase_key=phase, dst_rank=0
    )
    assert complete is True and complete_at == 3


def test_combine_expectations_use_combine_payload_spec_bytes() -> None:
    dispatch = PhaseKey("run", "sample", 0, PhaseKind.DISPATCH)
    combine = PhaseKey("run", "sample", 0, PhaseKind.COMBINE)
    system = build_formal_system()
    register_truth(
        system,
        phase=dispatch,
        src_rank=0,
        rows=(1, 2),
        dispatch_bytes=(4, 8),
        combine_bytes=(6, 12),
    )

    expectations = system.backend.register_combine_expectations_from_dispatch_truth(
        dispatch_phase_key=dispatch,
        combine_phase_key=combine,
        original_rank=0,
        created_at_ns=7,
    )
    assert tuple(item.total_expected_payload_bytes for item in expectations) == (6, 12)
    assert tuple(item.src_rank for item in expectations) == (0, 1)
    assert all(item.descriptor_digest_or_none is None for item in expectations)


def test_local_and_remote_assembly_jointly_gate_release_without_local_dataplane() -> None:
    dispatch = PhaseKey("run", "sample", 0, PhaseKind.DISPATCH)
    combine = PhaseKey("run", "sample", 0, PhaseKind.COMBINE)
    system = build_formal_system(local_assembly_cost_ns=1)
    for rank in range(2):
        system.backend.register_dispatch_compute_spec(
            dispatch_phase_key=dispatch,
            next_combine_phase_key=combine,
            rank_id=rank,
            dispatch_local_postprocess_ns=0,
            dispatch_release_to_combine_source_ready_ns=50,
        )
    register_truth(
        system,
        phase=dispatch,
        src_rank=0,
        rows=(1, 0),
        dispatch_bytes=(4, 0),
        combine_bytes=(6, 0),
    )
    register_truth(
        system,
        phase=dispatch,
        src_rank=1,
        rows=(1, 0),
        dispatch_bytes=(4, 0),
        combine_bytes=(6, 0),
    )
    for rank in range(2):
        system.backend.on_bootstrap_dispatch_local_path_complete(
            phase_key=dispatch, rank_id=rank, at_ns=0
        )
    drain_kernel(system.kernel)

    request_by_src = {
        request.src_rank: digest for digest, request in system.control.requests.items()
    }
    system.control.deliver(
        request_by_src[0], delivery_start_ns=1, delivered_at_ns=1
    )
    drain_kernel(system.kernel)
    system.control.deliver(
        request_by_src[1], delivery_start_ns=3, delivered_at_ns=3
    )

    remote_edge = next(
        edge
        for edge in system.receiver.expected_edges_for_destination(
            phase_key=dispatch, dst_rank=0
        )
        if edge.src_rank == 1
    )
    remote_task = CanonicalTransferTask(
        task_id="remote-1-to-0",
        edge_key=remote_edge.edge_key,
        phase_key=dispatch,
        src_rank=1,
        dst_rank=0,
        chunk_index=0,
        byte_offset=0,
        payload_bytes=4,
        expectation_digest=remote_edge.expectation_digest,
        taskization_digest="taskization-v1",
        registered_at_ns=3,
    )
    system.backend.register_canonical_task_catalogue((remote_task,))
    run_kernel_through(system.kernel, 5)

    local_edge = next(
        edge
        for edge in system.receiver.expected_edges_for_destination(
            phase_key=dispatch, dst_rank=0
        )
        if edge.src_rank == 0
    )
    assert local_edge.task_ids == []
    assert system.receiver.receive_permit(remote_task.task_id) is not None
    assert len(system.receiver.tasks_by_key) == 1
    assert system.backend.rank_release_at(phase_key=dispatch, rank_id=0) is None
    assert system.receiver.current_memory(0)["final_assembly_bytes"] == 4

    system.backend.on_transfer_completed(task_id=remote_task.task_id, at_ns=10)
    drain_kernel(system.kernel)

    assert system.backend.rank_release_at(phase_key=dispatch, rank_id=0) == 12
    assert system.receiver.current_memory(0)["staging_bytes"] == 0
    assert system.receiver.current_memory(0)["final_assembly_bytes"] == 0
    metrics = system.receiver.metrics_snapshot()
    assert metrics.peak_staging_bytes_per_rank[0] == 4
    assert metrics.peak_final_assembly_bytes_per_rank[0] == 8


def test_closed_combine_final_assembly_is_consumed_by_local_path() -> None:
    dispatch = PhaseKey("run", "sample", 0, PhaseKind.DISPATCH)
    combine = PhaseKey("run", "sample", 0, PhaseKind.COMBINE)
    next_dispatch = PhaseKey("run", "sample", 1, PhaseKind.DISPATCH)
    system = build_formal_system()
    register_truth(
        system,
        phase=dispatch,
        src_rank=0,
        rows=(1, 0),
        dispatch_bytes=(4, 0),
        combine_bytes=(6, 0),
    )
    register_truth(
        system,
        phase=next_dispatch,
        src_rank=0,
        rows=(1, 0),
        dispatch_bytes=(4, 0),
        combine_bytes=(6, 0),
    )
    system.backend.register_local_path_spec(
        combine_phase_key=combine,
        next_dispatch_phase_key=next_dispatch,
        rank_id=0,
        combine_release_to_router_ready_ns=0,
        router_and_pack_ns=0,
    )
    system.backend.register_combine_expectations_from_dispatch_truth(
        dispatch_phase_key=dispatch,
        combine_phase_key=combine,
        original_rank=0,
        created_at_ns=0,
    )
    system.backend.on_source_payload_ready(
        phase_key=combine, src_rank=0, at_ns=0
    )
    drain_kernel(system.kernel)

    assert system.receiver.current_memory(0)["final_assembly_bytes"] == 0
    snapshot = system.backend.combine_destination_snapshot(
        phase_key=combine, dst_rank=0
    )
    assert snapshot["data_ready_at_ns"] == 0
    assert snapshot["local_path_complete_at_ns"] == 0
    assert system.backend.exact_row_descriptor(
        phase_key=next_dispatch, src_rank=0
    ) is not None


def test_completion_adapter_applies_transport_completion_before_receiver() -> None:
    phase = PhaseKey("run", "sample", 0, PhaseKind.DISPATCH)
    order: list[str] = []

    task = CanonicalTransferTask(
        task_id="task-0",
        edge_key=EdgeKey(phase, 0, 1),
        phase_key=phase,
        src_rank=0,
        dst_rank=1,
        chunk_index=0,
        byte_offset=0,
        payload_bytes=4,
        expectation_digest="expectation-v1",
        taskization_digest="taskization-v1",
        registered_at_ns=0,
    )

    class Catalogue:
        def get(self, task_id: str):
            assert task_id == "task-0"
            return task

    class Authority:
        def mark_completed(self, phase_key, task_id, *, at_ns):
            assert phase_key == phase and task_id == "task-0" and at_ns == 9
            order.append("SCHEDULER_COMPLETED")

    class Backend:
        def on_transfer_completed(self, *, task_id, at_ns):
            assert task_id == "task-0" and at_ns == 9
            order.append("RECEIVER_COMPLETED")

    adapter = SchedulerBackendCompletionAdapter(
        authority=Authority(), catalogue=Catalogue(), backend=Backend()
    )
    adapter.on_transfer_completed(
        TransferCompleted(
            task_id="task-0",
            batch_id="batch-0",
            complete_at_ns=9,
            payload_bytes=4,
            physical_record_digest="physical-record-v1",
        )
    )
    assert order == ["SCHEDULER_COMPLETED", "RECEIVER_COMPLETED"]
