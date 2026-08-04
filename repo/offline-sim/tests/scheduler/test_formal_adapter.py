from __future__ import annotations

from dataclasses import dataclass

import pytest

from rs_sim import (
    LinkClass,
    ReceivePermit,
    SimulationKernel,
    SubmitOutcome,
    TaskResourceFootprint,
    make_control_plane_profile,
    make_hardware_profile,
    make_network_topology,
    make_transport_snapshot,
)
from tests.support.transport_conformance import TransportConformanceFake
from rs_sim.runtime.adapters.scheduler import build_scheduler_port_bundle
from rs_sim.scheduler.execution.compiler import (
    BatchCompiler,
    BatchValidator,
    CompilationError,
    ExecutionStabilizer,
    FormalTransportResourceAdapter,
)
from tests.scheduler.helpers import expectation, make_ready, phase, window


def _topology(world_size: int = 6):
    rank_to_node = tuple(0 if rank < world_size // 2 else 1 for rank in range(world_size))
    return make_network_topology(
        topology_id=f"topology-{world_size}",
        rank_to_node=rank_to_node,
        tx_nic_id_by_rank=tuple(f"tx-{rank}" for rank in range(world_size)),
        rx_nic_id_by_rank=tuple(f"rx-{rank}" for rank in range(world_size)),
        lane_ids_by_link_class=(
            (LinkClass.INTRA_NODE, ("intra-0", "intra-1", "intra-2")),
            (LinkClass.INTER_NODE, ("inter-0", "inter-1", "inter-2")),
        ),
        nic_id_by_lane=(
            ("intra-0", "lane-nic-i0"),
            ("intra-1", "lane-nic-i1"),
            ("intra-2", "lane-nic-i2"),
            ("inter-0", "lane-nic-e0"),
            ("inter-1", "lane-nic-e1"),
            ("inter-2", "lane-nic-e2"),
        ),
    )


class _Resolver:
    def __init__(self, topology, eligible_by_task=None):
        self.topology = topology
        self.eligible_by_task = eligible_by_task or {}

    def footprint(self, task):
        link_class = (
            LinkClass.INTRA_NODE
            if self.topology.rank_to_node[task.src_rank] == self.topology.rank_to_node[task.dst_rank]
            else LinkClass.INTER_NODE
        )
        lanes = self.eligible_by_task.get(
            task.task_id,
            dict(self.topology.lane_ids_by_link_class)[link_class],
        )
        return TaskResourceFootprint(
            task_id=task.task_id,
            topology_digest=self.topology.topology_digest,
            link_class=link_class,
            src_resource_id=f"rank-tx:{task.src_rank}",
            dst_resource_id=f"rank-rx:{task.dst_rank}",
            tx_nic_id=self.topology.tx_nic_id_by_rank[task.src_rank],
            rx_nic_id=self.topology.rx_nic_id_by_rank[task.dst_rank],
            eligible_lane_ids=tuple(lanes),
        )


def _snapshot(topology, *, hardware_digest="hardware", max_batch_tasks=4, busy_nics=()):
    return make_transport_snapshot(
        snapshot_at_ns=20,
        max_batch_tasks=max_batch_tasks,
        busy_src_ranks=(),
        busy_dst_ranks=(),
        busy_nic_ids=tuple(busy_nics),
        busy_lane_ids=(),
        available_lane_ids_by_link_class=topology.lane_ids_by_link_class,
        hardware_profile_digest=hardware_digest,
        topology_digest=topology.topology_digest,
    )


def _formal_compiler(stack, resolver, *, hardware_digest="hardware"):
    from rs_sim.runtime.adapters.scheduler import SchedulerTaskLookupAdapter

    resources = FormalTransportResourceAdapter(
        task_lookup=SchedulerTaskLookupAdapter(stack.catalogue),
        resource_resolver=resolver,
        expected_hardware_profile_digest=hardware_digest,
    )
    compiler = BatchCompiler(
        catalogue=stack.catalogue,
        runtime=stack.runtime,
        authority=stack.authority,
        resources=resources,
    )
    validator = BatchValidator(
        catalogue=stack.catalogue,
        runtime=stack.runtime,
        authority=stack.authority,
        resources=resources,
    )
    return resources, compiler, validator


def test_formal_resource_adapter_binds_shared_resolver_and_profile(stack):
    phase_key = phase()
    for src, dst in ((0, 3), (1, 4), (2, 5)):
        stack.controller.register_expectation(
            expectation(phase_key, src, dst, 64), registered_at_ns=10 + src
        )
    task_ids = stack.catalogue.task_ids_for_phase(phase_key)
    make_ready(stack, task_ids)
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(), ordered_task_ids=task_ids, now_ns=20
    )

    topology = _topology()
    sorted_ids = sorted(task_ids)
    resolver = _Resolver(
        topology,
        eligible_by_task={
            sorted_ids[0]: ("inter-0",),
            sorted_ids[1]: ("inter-1", "inter-2"),
            sorted_ids[2]: ("inter-0", "inter-1"),
        },
    )
    _, compiler, validator = _formal_compiler(stack, resolver)
    snapshot = _snapshot(topology)
    attempt = compiler.compile_next(phase_key=phase_key, snapshot=snapshot, now_ns=21)
    assert attempt.code == "BATCH_READY"
    assert set(attempt.batch.task_ids) == set(task_ids)
    assert validator.validate(attempt.batch, snapshot=snapshot, now_ns=21).code == "ACCEPTED"

    wrong_hardware = _snapshot(topology, hardware_digest="wrong")
    with pytest.raises(CompilationError, match="HARDWARE_PROFILE"):
        compiler.compile_next(phase_key=phase_key, snapshot=wrong_hardware, now_ns=22)

    other_topology = _topology(4)
    wrong_topology = make_transport_snapshot(
        snapshot_at_ns=20,
        max_batch_tasks=4,
        busy_src_ranks=(),
        busy_dst_ranks=(),
        busy_nic_ids=(),
        busy_lane_ids=(),
        available_lane_ids_by_link_class=other_topology.lane_ids_by_link_class,
        hardware_profile_digest="hardware",
        topology_digest=other_topology.topology_digest,
    )
    with pytest.raises(CompilationError, match="TOPOLOGY"):
        compiler.compile_next(phase_key=phase_key, snapshot=wrong_topology, now_ns=22)


def _permit(task, *, expectation_digest=None):
    return ReceivePermit(
        permit_id=f"permit:{task.task_id}",
        task_id=task.task_id,
        edge_key=task.edge_key,
        chunk_index=task.chunk_index,
        byte_offset=task.byte_offset,
        task_bytes=task.payload_bytes,
        credit_reservation_id=f"credit:{task.task_id}",
        expectation_digest=expectation_digest or task.expectation_digest,
        descriptor_digest_or_none="descriptor",
        posted_at_ns=15,
    )


@dataclass
class _PermitLookup:
    permits: dict[str, ReceivePermit]

    def permit(self, task_id: str):
        return self.permits.get(task_id)


@dataclass
class _Backend:
    completed: list[tuple[str, int]]

    def on_transfer_completed(self, *, task_id: str, at_ns: int):
        self.completed.append((task_id, at_ns))


@dataclass
class _Bridge:
    releases: list[object]

    def notify_transport_resource_release(self, phase_key):
        self.releases.append(phase_key)


def _profiles():
    hardware = make_hardware_profile(
        profile_id="synthetic-hw",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        max_batch_tasks=2,
        launch_delay_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 1),
            (LinkClass.INTER_NODE, 1),
        ),
        fixed_latency_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 2),
            (LinkClass.INTER_NODE, 2),
        ),
        bandwidth_bytes_per_second_by_link_class=(
            (LinkClass.INTRA_NODE, 1_000_000_000),
            (LinkClass.INTER_NODE, 1_000_000_000),
        ),
    )
    control = make_control_plane_profile(
        profile_id="synthetic-control",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        fixed_latency_ns=1,
        bandwidth_bytes_per_second=1_000_000_000,
    )
    return hardware, control


def _formal_transaction_stack(stack, *, bad_expectation=False):
    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 3, 64), registered_at_ns=10
    )
    task_id = stack.catalogue.task_ids_for_phase(phase_key)[0]
    make_ready(stack, (task_id,))
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(), ordered_task_ids=(task_id,), now_ns=20
    )
    task = stack.catalogue.get(task_id)
    topology = _topology()
    resolver = _Resolver(topology)
    hardware, control = _profiles()
    backend = _Backend([])
    bridge = _Bridge([])
    bundle = build_scheduler_port_bundle(
        catalogue=stack.catalogue,
        authority=stack.authority,
        backend=backend,
        scheduler_bridge=bridge,
        resource_resolver=resolver,
        expected_hardware_profile_digest=hardware.profile_digest,
    )
    permit = _permit(
        task,
        expectation_digest=("wrong-expectation" if bad_expectation else task.expectation_digest),
    )
    kernel = SimulationKernel()
    transport = TransportConformanceFake(
        kernel=kernel,
        task_lookup=bundle.task_lookup,
        permit_lookup=_PermitLookup({task_id: permit}),
        authority_validation=bundle.authority_validation,
        resource_resolver=resolver,
        completion_sink=bundle.completion_sink,
        resource_release_sink=bundle.resource_release_sink,
        hardware_profile=hardware,
        control_profile=control,
    )
    compiler = BatchCompiler(
        catalogue=stack.catalogue,
        runtime=stack.runtime,
        authority=stack.authority,
        resources=bundle.resource_adapter,
    )
    validator = BatchValidator(
        catalogue=stack.catalogue,
        runtime=stack.runtime,
        authority=stack.authority,
        resources=bundle.resource_adapter,
    )
    stabilizer = ExecutionStabilizer(
        compiler=compiler,
        validator=validator,
        authority=stack.authority,
    )
    return phase_key, task_id, backend, bridge, kernel, transport, stabilizer


def test_wrong_permit_expectation_binding_fails_closed(stack):
    phase_key, task_id, _, _, kernel, transport, stabilizer = _formal_transaction_stack(
        stack, bad_expectation=True
    )
    with pytest.raises(CompilationError, match="fatal result"):
        stabilizer.stabilize(
            phase_key=phase_key,
            snapshot_provider=transport.snapshot,
            transport=transport,
            now_ns=21,
        )
    assert stack.runtime.facts(task_id).state == "READY_UNCOMMITTED"
    assert kernel.pending_event_count() == 0


def test_confirm_is_first_point_that_can_create_start_event(stack):
    phase_key, task_id, backend, bridge, kernel, base_transport, stabilizer = _formal_transaction_stack(
        stack, bad_expectation=False
    )

    class ObservedConfirmTransport:
        def snapshot(self):
            return base_transport.snapshot()

        def prepare_commit(self, batch, commit_time_ns):
            return base_transport.prepare_commit(batch, commit_time_ns)

        def abort_commit(self, receipt):
            return base_transport.abort_commit(receipt)

        def confirm_commit(self, receipt):
            assert stack.runtime.facts(task_id).state == "COMMITTED"
            assert kernel.pending_event_count() == 0
            base_transport.confirm_commit(receipt)
            assert kernel.pending_event_count() > 0

    transport = ObservedConfirmTransport()
    result = stabilizer.stabilize(
        phase_key=phase_key,
        snapshot_provider=transport.snapshot,
        transport=transport,
        now_ns=21,
    )
    assert result.accepted_task_ids == (task_id,)
    assert stack.runtime.facts(task_id).state == "COMMITTED"
    kernel.run_until_complete(
        lambda: stack.runtime.facts(task_id).state == "COMPLETED"
    )
    assert stack.runtime.facts(task_id).state == "COMPLETED"
    assert backend.completed and backend.completed[0][0] == task_id
    assert bridge.releases == [phase_key]
