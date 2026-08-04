from __future__ import annotations

from dataclasses import dataclass

import pytest

from rs_sim import (
    CanonicalTransferTask,
    EdgeKey,
    ExpectationOrigin,
    LinkClass,
    PhaseKey,
    PhaseKind,
    ReceiveExpectation,
    ReceivePermit,
    SimulationKernel,
    SubmitOutcome,
    make_authority_stamp,
    make_control_plane_profile,
    make_exact_row_descriptor,
    make_hardware_profile,
    make_network_topology,
    make_task_resource_footprint,
    make_transfer_batch,
    stable_digest,
)
from tests.support.transport_conformance import TransportConformanceFake
from rs_sim.runtime.adapters.public_ports import SharedTopologyTaskResolver


@dataclass
class Lookup:
    value: object

    def task(self, task_id: str):
        assert task_id == self.value.task_id
        return self.value


@dataclass
class PermitLookup:
    value: ReceivePermit | None

    def permit(self, task_id: str):
        if self.value is None:
            return None
        assert task_id == self.value.task_id
        return self.value


@dataclass
class Authority:
    current: object

    def authority_is_current(self, *, phase_key, authority_stamp):
        return phase_key == self.phase_key and authority_stamp == self.current

    @property
    def phase_key(self):
        return PHASE


class CompletionLog:
    def __init__(self):
        self.started = []
        self.completed = []

    def on_transfer_started(self, event):
        self.started.append(event)

    def on_transfer_completed(self, event):
        self.completed.append(event)


class ReleaseLog:
    def __init__(self):
        self.phases = []

    def on_transport_resources_released(self, phase_key):
        self.phases.append(phase_key)


class ControlLog:
    def __init__(self):
        self.deliveries = []

    def on_control_plane_delivery(self, delivery):
        self.deliveries.append(delivery)


PHASE = PhaseKey("run", "public-contract", 1, PhaseKind.DISPATCH)
EDGE = EdgeKey(PHASE, 0, 1)
EXPECTATION_DIGEST = stable_digest(("expectation", EDGE), domain="TEST_EXPECTATION")


def make_task_and_permit(*, permit_expectation_digest: str = EXPECTATION_DIGEST):
    task = CanonicalTransferTask(
        task_id="task:public",
        edge_key=EDGE,
        phase_key=PHASE,
        src_rank=0,
        dst_rank=1,
        chunk_index=0,
        byte_offset=0,
        payload_bytes=8,
        expectation_digest=EXPECTATION_DIGEST,
        taskization_digest=stable_digest((EXPECTATION_DIGEST, 0, 8), domain="TASKIZATION"),
        registered_at_ns=3,
    )
    permit = ReceivePermit(
        permit_id="permit:public",
        task_id=task.task_id,
        edge_key=EDGE,
        chunk_index=0,
        byte_offset=0,
        task_bytes=8,
        credit_reservation_id="credit:public",
        expectation_digest=permit_expectation_digest,
        descriptor_digest_or_none="descriptor:public",
        posted_at_ns=4,
    )
    return task, permit


def topology():
    return make_network_topology(
        topology_id="two-node",
        rank_to_node=(0, 1),
        tx_nic_id_by_rank=("node0-tx", "node1-tx"),
        rx_nic_id_by_rank=("node0-rx", "node1-rx"),
        lane_ids_by_link_class=(
            (LinkClass.INTRA_NODE, ("intra0",)),
            (LinkClass.INTER_NODE, ("inter0",)),
        ),
        nic_id_by_lane=(("intra0", "fabric-intra"), ("inter0", "fabric-inter")),
    )


def profiles():
    hardware = make_hardware_profile(
        profile_id="synthetic-data",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        max_batch_tasks=2,
        launch_delay_ns_by_link_class=((LinkClass.INTRA_NODE, 2), (LinkClass.INTER_NODE, 3)),
        fixed_latency_ns_by_link_class=((LinkClass.INTRA_NODE, 5), (LinkClass.INTER_NODE, 7)),
        bandwidth_bytes_per_second_by_link_class=((LinkClass.INTRA_NODE, 1_000_000_000), (LinkClass.INTER_NODE, 500_000_000)),
    )
    control = make_control_plane_profile(
        profile_id="synthetic-control",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        fixed_latency_ns=5,
        bandwidth_bytes_per_second=1_000_000_000,
    )
    return hardware, control


def build_transport(*, permit_digest: str = EXPECTATION_DIGEST):
    kernel = SimulationKernel()
    task, permit = make_task_and_permit(permit_expectation_digest=permit_digest)
    stamp = make_authority_stamp(phase_token="phase-token", plan_id="plan-1", phase_plan_epoch=2)
    completion = CompletionLog()
    release = ReleaseLog()
    hardware, control = profiles()
    topo = topology()
    transport = TransportConformanceFake(
        kernel=kernel,
        task_lookup=Lookup(task),
        permit_lookup=PermitLookup(permit),
        authority_validation=Authority(stamp),
        resource_resolver=SharedTopologyTaskResolver(topo),
        completion_sink=completion,
        resource_release_sink=release,
        hardware_profile=hardware,
        control_profile=control,
    )
    batch = make_transfer_batch(
        batch_id="batch-1",
        phase_key=PHASE,
        task_ids=(task.task_id,),
        authority_stamp=stamp,
        link_class=LinkClass.INTER_NODE,
        topology_digest=topo.topology_digest,
        compiled_at_ns=10,
    )
    return kernel, transport, batch, completion, release, topo


def test_content_only_descriptor_identity_excludes_publication_time():
    kwargs = dict(
        phase_key=PHASE,
        src_rank=0,
        realized_rows_by_destination=(0, 2),
        payload_bytes_by_destination=(0, 8),
        payload_spec_digest="payload-spec",
        descriptor_payload_bytes=24,
    )
    first = make_exact_row_descriptor(**kwargs, published_at_ns=5)
    second = make_exact_row_descriptor(**kwargs, published_at_ns=15)
    assert first.descriptor_digest == second.descriptor_digest
    expectation_one = ReceiveExpectation(
        edge_key=EDGE, phase_key=PHASE, src_rank=0, dst_rank=1,
        total_expected_payload_bytes=8,
        expectation_digest=stable_digest((EDGE, first.descriptor_digest, 8), domain="EXPECTATION"),
        origin=ExpectationOrigin.DISPATCH_DESCRIPTOR, created_at_ns=20,
        zero_edge=False, descriptor_digest_or_none=first.descriptor_digest,
    )
    expectation_two = ReceiveExpectation(
        edge_key=EDGE, phase_key=PHASE, src_rank=0, dst_rank=1,
        total_expected_payload_bytes=8,
        expectation_digest=stable_digest((EDGE, second.descriptor_digest, 8), domain="EXPECTATION"),
        origin=ExpectationOrigin.DISPATCH_DESCRIPTOR, created_at_ns=30,
        zero_edge=False, descriptor_digest_or_none=second.descriptor_digest,
    )
    assert expectation_one.expectation_digest == expectation_two.expectation_digest

    from tests.scheduler.conftest import build_stack
    first_stack = build_stack(chunk_bytes=4, alignment_bytes=1)
    second_stack = build_stack(chunk_bytes=4, alignment_bytes=1)
    first_tasks = first_stack.controller.register_expectation(expectation_one, registered_at_ns=5)
    second_tasks = second_stack.controller.register_expectation(expectation_two, registered_at_ns=15)
    assert tuple(task.task_id for task in first_tasks) == tuple(task.task_id for task in second_tasks)
    assert first_stack.catalogue.digest() == second_stack.catalogue.digest()


def test_public_transport_constructs_receipt_without_scheduler_private_state_and_confirm_is_event_boundary():
    kernel, transport, batch, completion, release, _ = build_transport()
    outcome, receipt = transport.prepare_commit(batch, 20)
    assert outcome is SubmitOutcome.PREPARED
    assert receipt is not None
    assert receipt.authority_stamp == batch.authority_stamp
    assert receipt.plan_id == batch.plan_id
    assert receipt.phase_plan_epoch == batch.phase_plan_epoch
    assert kernel.pending_event_count() == 0
    transport.confirm_commit(receipt)
    assert kernel.pending_event_count() == 1
    
    while kernel.has_events():
        kernel.run_next_timestamp()
    assert len(completion.started) == len(completion.completed) == 1
    assert release.phases == [PHASE]


def test_expectation_digest_mismatch_is_fatal_without_side_effects():
    kernel, transport, batch, _, _, _ = build_transport(permit_digest="wrong-expectation")
    before = transport.snapshot()
    outcome, receipt = transport.prepare_commit(batch, 20)
    assert outcome is SubmitOutcome.FATAL_CONTRACT_ERROR
    assert receipt is None
    assert transport.prepared_count == 0
    assert kernel.pending_event_count() == 0
    assert transport.snapshot() == before


def test_topology_digest_mismatch_is_fatal_without_side_effects():
    kernel, transport, batch, _, _, _ = build_transport()
    bad = make_transfer_batch(
        batch_id=batch.batch_id,
        phase_key=batch.phase_key,
        task_ids=batch.task_ids,
        authority_stamp=batch.authority_stamp,
        link_class=batch.link_class,
        topology_digest="wrong-topology",
        compiled_at_ns=batch.compiled_at_ns,
    )
    outcome, receipt = transport.prepare_commit(bad, 20)
    assert outcome is SubmitOutcome.FATAL_CONTRACT_ERROR
    assert receipt is None
    assert kernel.pending_event_count() == 0


def test_control_plane_delivery_uses_public_sink_and_fifo_profile():
    kernel, transport, _, _, _, _ = build_transport()
    sink = ControlLog()
    transport.attach_delivery_sink(sink)
    first = make_exact_row_descriptor(
        phase_key=PHASE, src_rank=0, realized_rows_by_destination=(0, 2),
        payload_bytes_by_destination=(0, 8), payload_spec_digest="spec",
        published_at_ns=3, descriptor_payload_bytes=10,
    )
    second = make_exact_row_descriptor(
        phase_key=PHASE, src_rank=1, realized_rows_by_destination=(1, 0),
        payload_bytes_by_destination=(4, 0), payload_spec_digest="spec",
        published_at_ns=3, descriptor_payload_bytes=10,
    )
    from rs_sim import make_row_broadcast_request
    transport.publish_row(make_row_broadcast_request(first))
    transport.publish_row(make_row_broadcast_request(second))
    
    while kernel.has_events():
        kernel.run_next_timestamp()
    assert len(sink.deliveries) == 2
    assert sink.deliveries[0].delivered_at_ns <= sink.deliveries[1].delivery_start_ns


def test_resource_footprint_uses_single_shared_topology_truth():
    task, _ = make_task_and_permit()
    topo = topology()
    footprint = make_task_resource_footprint(
        task_id=task.task_id, src_rank=task.src_rank, dst_rank=task.dst_rank, topology=topo
    )
    assert footprint.topology_digest == topo.topology_digest
    assert footprint.link_class is LinkClass.INTER_NODE
    assert footprint.tx_nic_id == "node0-tx"
    assert footprint.rx_nic_id == "node1-rx"
