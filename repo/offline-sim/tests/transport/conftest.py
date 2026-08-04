from __future__ import annotations

from dataclasses import dataclass

import pytest

from rs_sim import (
    CanonicalTransferTask,
    EdgeKey,
    LinkClass,
    PhaseKey,
    PhaseKind,
    ReceivePermit,
    SimulationKernel,
    make_authority_stamp,
    make_hardware_profile,
    make_network_topology,
    make_task_resource_footprint,
    make_transfer_batch,
    stable_digest,
)
from rs_sim.transport import FormalDataPlaneTransport


@dataclass
class TaskLookup:
    tasks: dict[str, CanonicalTransferTask]

    def task(self, task_id: str) -> CanonicalTransferTask:
        return self.tasks[task_id]


@dataclass
class PermitLookup:
    permits: dict[str, ReceivePermit]

    def permit(self, task_id: str) -> ReceivePermit | None:
        return self.permits.get(task_id)


@dataclass
class AuthorityValidation:
    phase_key: PhaseKey
    stamp: object
    current: bool = True

    def authority_is_current(self, *, phase_key, authority_stamp) -> bool:
        return self.current and phase_key == self.phase_key and authority_stamp == self.stamp


class SharedResolver:
    def __init__(self, topology, overrides=None):
        self._topology = topology
        self.overrides = overrides or {}

    @property
    def topology(self):
        return self._topology

    def footprint(self, task):
        return self.overrides.get(
            task.task_id,
            make_task_resource_footprint(
                task_id=task.task_id,
                src_rank=task.src_rank,
                dst_rank=task.dst_rank,
                topology=self._topology,
            ),
        )


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


def topology_four_ranks():
    return make_network_topology(
        topology_id="transport-four-ranks",
        rank_to_node=(0, 1, 2, 3),
        tx_nic_id_by_rank=("n0-tx", "n1-tx", "n2-tx", "n3-tx"),
        rx_nic_id_by_rank=("n0-rx", "n1-rx", "n2-rx", "n3-rx"),
        lane_ids_by_link_class=(
            (LinkClass.INTRA_NODE, ("intra-0", "intra-1")),
            (LinkClass.INTER_NODE, ("inter-0", "inter-1")),
        ),
        nic_id_by_lane=(
            ("intra-0", "fabric-intra"),
            ("intra-1", "fabric-intra"),
            ("inter-0", "fabric-inter"),
            ("inter-1", "fabric-inter"),
        ),
    )


def topology_mixed_nodes():
    return make_network_topology(
        topology_id="transport-mixed-nodes",
        rank_to_node=(0, 0, 1, 2),
        tx_nic_id_by_rank=("node0-tx", "node0-tx", "node1-tx", "node2-tx"),
        rx_nic_id_by_rank=("node0-rx", "node0-rx", "node1-rx", "node2-rx"),
        lane_ids_by_link_class=(
            (LinkClass.INTRA_NODE, ("intra-0", "intra-1")),
            (LinkClass.INTER_NODE, ("inter-0", "inter-1")),
        ),
        nic_id_by_lane=(
            ("intra-0", "fabric-intra"),
            ("intra-1", "fabric-intra"),
            ("inter-0", "fabric-inter"),
            ("inter-1", "fabric-inter"),
        ),
    )


def hardware_profile(max_batch_tasks=4):
    return make_hardware_profile(
        profile_id="transport-test-profile",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        max_batch_tasks=max_batch_tasks,
        launch_delay_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 2),
            (LinkClass.INTER_NODE, 3),
        ),
        fixed_latency_ns_by_link_class=(
            (LinkClass.INTRA_NODE, 5),
            (LinkClass.INTER_NODE, 7),
        ),
        bandwidth_bytes_per_second_by_link_class=(
            (LinkClass.INTRA_NODE, 2_000_000_000),
            (LinkClass.INTER_NODE, 1_000_000_000),
        ),
    )


def make_task(
    phase_key: PhaseKey,
    *,
    task_id: str,
    src: int,
    dst: int,
    payload_bytes: int = 8,
    chunk_index: int = 0,
    byte_offset: int = 0,
    expectation_digest: str | None = None,
):
    edge = EdgeKey(phase_key, src, dst)
    expectation = expectation_digest or stable_digest(
        (edge, payload_bytes), domain="TRANSPORT_TEST_EXPECTATION"
    )
    task = CanonicalTransferTask(
        task_id=task_id,
        edge_key=edge,
        phase_key=phase_key,
        src_rank=src,
        dst_rank=dst,
        chunk_index=chunk_index,
        byte_offset=byte_offset,
        payload_bytes=payload_bytes,
        expectation_digest=expectation,
        taskization_digest=stable_digest(
            (expectation, chunk_index, byte_offset, payload_bytes),
            domain="TRANSPORT_TEST_TASKIZATION",
        ),
        registered_at_ns=1,
    )
    permit = ReceivePermit(
        permit_id=f"permit:{task_id}",
        task_id=task_id,
        edge_key=edge,
        chunk_index=chunk_index,
        byte_offset=byte_offset,
        task_bytes=payload_bytes,
        credit_reservation_id=f"credit:{task_id}",
        expectation_digest=expectation,
        descriptor_digest_or_none=(
            f"descriptor:{task_id}"
            if phase_key.phase_kind is PhaseKind.DISPATCH
            else None
        ),
        posted_at_ns=2,
    )
    return task, permit


@dataclass
class DataHarness:
    kernel: SimulationKernel
    transport: FormalDataPlaneTransport
    tasks: dict[str, CanonicalTransferTask]
    permits: dict[str, ReceivePermit]
    stamp: object
    phase: PhaseKey
    completion: CompletionLog
    release: ReleaseLog
    resolver: SharedResolver

    def batch(self, *task_ids: str, link_class=LinkClass.INTER_NODE, topology_digest=None):
        return make_transfer_batch(
            batch_id="batch:" + "+".join(task_ids),
            phase_key=self.phase,
            task_ids=tuple(task_ids),
            authority_stamp=self.stamp,
            link_class=link_class,
            topology_digest=topology_digest or self.resolver.topology.topology_digest,
            compiled_at_ns=0,
        )


def build_harness(*, topology=None, task_specs=None, resolver_overrides=None, profile=None, bandwidth_contention=None):
    topology = topology or topology_four_ranks()
    phase = PhaseKey("run", "transport-test", 0, PhaseKind.DISPATCH)
    task_specs = task_specs or (
        ("t0", 0, 2, 8),
        ("t1", 1, 3, 16),
    )
    tasks = {}
    permits = {}
    for task_id, src, dst, payload in task_specs:
        task, permit = make_task(
            phase, task_id=task_id, src=src, dst=dst, payload_bytes=payload
        )
        tasks[task_id] = task
        permits[task_id] = permit
    stamp = make_authority_stamp(
        phase_token="opaque-token", plan_id="plan-7", phase_plan_epoch=9
    )
    kernel = SimulationKernel()
    completion = CompletionLog()
    release = ReleaseLog()
    resolver = SharedResolver(topology, resolver_overrides)
    transport = FormalDataPlaneTransport(
        kernel=kernel,
        task_lookup=TaskLookup(tasks),
        permit_lookup=PermitLookup(permits),
        authority_validation=AuthorityValidation(phase, stamp),
        resource_resolver=resolver,
        completion_sink=completion,
        resource_release_sink=release,
        hardware_profile=profile or hardware_profile(),
        bandwidth_contention=bandwidth_contention,
    )
    return DataHarness(
        kernel, transport, tasks, permits, stamp, phase, completion, release, resolver
    )


@pytest.fixture
def harness():
    return build_harness()
