from __future__ import annotations

"""Canonical factories for cross-module immutable contracts.

All digest formulas live here so backend, scheduler, transport, and runtime cannot
silently invent incompatible identities.  Authoritative inputs reject floats
through :func:`rs_sim.contracts.digest.stable_digest`.
"""

from .schema import (
    AuthorityStamp,
    CommitReceipt,
    ControlPlaneProfile,
    ExactDispatchRowTruth,
    ExactRowDescriptor,
    HardwareProfile,
    LinkClass,
    NetworkTopology,
    PhaseKey,
    RowBroadcastRequest,
    TaskResourceFootprint,
    TransferBatch,
    TransportSnapshot,
)
from rs_sim.contracts.digest import stable_digest

CONTRACT_SCHEMA_VERSION = "RS_SIM_CANONICAL_TASKIZATION"


def ceil_transfer_time_ns(payload_bytes: int, bandwidth_bytes_per_second: int) -> int:
    """Return ceil(payload / bandwidth) in integer nanoseconds."""

    if not isinstance(payload_bytes, int) or isinstance(payload_bytes, bool) or payload_bytes < 0:
        raise ValueError("payload_bytes must be a non-negative int")
    if (
        not isinstance(bandwidth_bytes_per_second, int)
        or isinstance(bandwidth_bytes_per_second, bool)
        or bandwidth_bytes_per_second <= 0
    ):
        raise ValueError("bandwidth_bytes_per_second must be a positive int")
    return (payload_bytes * 1_000_000_000 + bandwidth_bytes_per_second - 1) // bandwidth_bytes_per_second


def make_authority_stamp(
    *, phase_token: str, plan_id: str, phase_plan_epoch: int
) -> AuthorityStamp:
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "phase_token": str(phase_token),
        "plan_id": str(plan_id),
        "phase_plan_epoch": int(phase_plan_epoch),
    }
    return AuthorityStamp(
        phase_token=str(phase_token),
        plan_id=str(plan_id),
        phase_plan_epoch=int(phase_plan_epoch),
        authority_digest=stable_digest(payload, domain="AUTHORITY_STAMP"),
    )


def descriptor_truth_digest(
    *,
    phase_key: PhaseKey,
    src_rank: int,
    realized_rows_by_destination: tuple[int, ...],
    payload_bytes_by_destination: tuple[int, ...],
    payload_spec_digest: str,
    descriptor_payload_bytes: int,
) -> str:
    """Content-only descriptor digest; publication/delivery time is excluded."""

    return stable_digest(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "phase_key": phase_key,
            "src_rank": int(src_rank),
            "realized_rows_by_destination": tuple(int(v) for v in realized_rows_by_destination),
            "payload_bytes_by_destination": tuple(int(v) for v in payload_bytes_by_destination),
            "payload_spec_digest": str(payload_spec_digest),
            "descriptor_payload_bytes": int(descriptor_payload_bytes),
        },
        domain="EXACT_ROW_DESCRIPTOR_TRUTH",
    )


def make_exact_row_descriptor(
    *,
    phase_key: PhaseKey,
    src_rank: int,
    realized_rows_by_destination: tuple[int, ...],
    payload_bytes_by_destination: tuple[int, ...],
    payload_spec_digest: str,
    published_at_ns: int,
    descriptor_payload_bytes: int,
) -> ExactRowDescriptor:
    digest = descriptor_truth_digest(
        phase_key=phase_key,
        src_rank=src_rank,
        realized_rows_by_destination=realized_rows_by_destination,
        payload_bytes_by_destination=payload_bytes_by_destination,
        payload_spec_digest=payload_spec_digest,
        descriptor_payload_bytes=descriptor_payload_bytes,
    )
    return ExactRowDescriptor(
        phase_key=phase_key,
        src_rank=int(src_rank),
        realized_rows_by_destination=tuple(int(v) for v in realized_rows_by_destination),
        payload_bytes_by_destination=tuple(int(v) for v in payload_bytes_by_destination),
        payload_spec_digest=str(payload_spec_digest),
        descriptor_digest=digest,
        published_at_ns=int(published_at_ns),
        descriptor_payload_bytes=int(descriptor_payload_bytes),
    )


def make_exact_dispatch_row_truth(
    *,
    phase_key: PhaseKey,
    src_rank: int,
    realized_rows_by_destination: tuple[int, ...],
    dispatch_payload_bytes_by_destination: tuple[int, ...],
    combine_return_payload_bytes_by_expert: tuple[int, ...],
    dispatch_payload_spec_digest: str,
    combine_payload_spec_digest: str,
    descriptor_payload_bytes: int,
) -> ExactDispatchRowTruth:
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "phase_key": phase_key,
        "src_rank": int(src_rank),
        "realized_rows_by_destination": tuple(int(v) for v in realized_rows_by_destination),
        "dispatch_payload_bytes_by_destination": tuple(int(v) for v in dispatch_payload_bytes_by_destination),
        "combine_return_payload_bytes_by_expert": tuple(int(v) for v in combine_return_payload_bytes_by_expert),
        "dispatch_payload_spec_digest": str(dispatch_payload_spec_digest),
        "combine_payload_spec_digest": str(combine_payload_spec_digest),
        "descriptor_payload_bytes": int(descriptor_payload_bytes),
    }
    return ExactDispatchRowTruth(
        phase_key=phase_key,
        src_rank=int(src_rank),
        realized_rows_by_destination=payload["realized_rows_by_destination"],
        dispatch_payload_bytes_by_destination=payload["dispatch_payload_bytes_by_destination"],
        combine_return_payload_bytes_by_expert=payload["combine_return_payload_bytes_by_expert"],
        dispatch_payload_spec_digest=str(dispatch_payload_spec_digest),
        combine_payload_spec_digest=str(combine_payload_spec_digest),
        descriptor_payload_bytes=int(descriptor_payload_bytes),
        truth_digest=stable_digest(payload, domain="EXACT_DISPATCH_ROW_TRUTH"),
    )


def make_network_topology(
    *,
    topology_id: str,
    rank_to_node: tuple[int, ...],
    tx_nic_id_by_rank: tuple[str, ...],
    rx_nic_id_by_rank: tuple[str, ...],
    lane_ids_by_link_class: tuple[tuple[LinkClass, tuple[str, ...]], ...],
    nic_id_by_lane: tuple[tuple[str, str], ...],
) -> NetworkTopology:
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "topology_id": str(topology_id),
        "rank_to_node": tuple(int(v) for v in rank_to_node),
        "tx_nic_id_by_rank": tuple(str(v) for v in tx_nic_id_by_rank),
        "rx_nic_id_by_rank": tuple(str(v) for v in rx_nic_id_by_rank),
        "lane_ids_by_link_class": lane_ids_by_link_class,
        "nic_id_by_lane": nic_id_by_lane,
    }
    return NetworkTopology(
        topology_id=str(topology_id),
        topology_digest=stable_digest(payload, domain="NETWORK_TOPOLOGY"),
        rank_to_node=payload["rank_to_node"],
        tx_nic_id_by_rank=payload["tx_nic_id_by_rank"],
        rx_nic_id_by_rank=payload["rx_nic_id_by_rank"],
        lane_ids_by_link_class=lane_ids_by_link_class,
        nic_id_by_lane=nic_id_by_lane,
    )


def make_task_resource_footprint(
    *, task_id: str, src_rank: int, dst_rank: int, topology: NetworkTopology
) -> TaskResourceFootprint:
    if src_rank == dst_rank:
        raise ValueError("local diagonal work has no DataPlane resource footprint")
    if src_rank < 0 or dst_rank < 0 or src_rank >= topology.world_size or dst_rank >= topology.world_size:
        raise ValueError("task rank outside topology world_size")
    link_class = (
        LinkClass.INTRA_NODE
        if topology.rank_to_node[src_rank] == topology.rank_to_node[dst_rank]
        else LinkClass.INTER_NODE
    )
    lane_map = dict(topology.lane_ids_by_link_class)
    lanes = lane_map.get(link_class, ())
    if not lanes:
        raise ValueError(f"topology has no lanes for {link_class.value}")
    return TaskResourceFootprint(
        task_id=str(task_id),
        topology_digest=topology.topology_digest,
        link_class=link_class,
        src_resource_id=f"rank-tx:{src_rank}",
        dst_resource_id=f"rank-rx:{dst_rank}",
        tx_nic_id=topology.tx_nic_id_by_rank[src_rank],
        rx_nic_id=topology.rx_nic_id_by_rank[dst_rank],
        eligible_lane_ids=tuple(lanes),
    )


def make_transfer_batch(
    *,
    batch_id: str,
    phase_key: PhaseKey,
    task_ids: tuple[str, ...],
    authority_stamp: AuthorityStamp,
    link_class: LinkClass,
    topology_digest: str,
    compiled_at_ns: int,
) -> TransferBatch:
    semantic = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "batch_id": str(batch_id),
        "phase_key": phase_key,
        "task_ids": tuple(str(v) for v in task_ids),
        "authority_stamp": authority_stamp,
        "link_class": link_class,
        "topology_digest": str(topology_digest),
        "compiled_at_ns": int(compiled_at_ns),
    }
    return TransferBatch(
        batch_id=str(batch_id),
        batch_digest=stable_digest(semantic, domain="TRANSFER_BATCH"),
        phase_key=phase_key,
        task_ids=semantic["task_ids"],
        authority_stamp=authority_stamp,
        link_class=link_class,
        topology_digest=str(topology_digest),
        compiled_at_ns=int(compiled_at_ns),
    )


def make_row_broadcast_request(descriptor: ExactRowDescriptor) -> RowBroadcastRequest:
    return RowBroadcastRequest(
        phase_key=descriptor.phase_key,
        src_rank=descriptor.src_rank,
        realized_rows_by_destination=descriptor.realized_rows_by_destination,
        payload_bytes_by_destination=descriptor.payload_bytes_by_destination,
        payload_spec_digest=descriptor.payload_spec_digest,
        descriptor_digest=descriptor.descriptor_digest,
        published_at_ns=descriptor.published_at_ns,
        descriptor_payload_bytes=descriptor.descriptor_payload_bytes,
    )



def make_hardware_profile(
    *,
    profile_id: str,
    profile_provenance: str,
    performance_eligible: bool,
    max_batch_tasks: int,
    launch_delay_ns_by_link_class: tuple[tuple[LinkClass, int], ...],
    fixed_latency_ns_by_link_class: tuple[tuple[LinkClass, int], ...],
    bandwidth_bytes_per_second_by_link_class: tuple[tuple[LinkClass, int], ...],
) -> HardwareProfile:
    semantic = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "profile_id": str(profile_id),
        "profile_provenance": str(profile_provenance),
        "performance_eligible": bool(performance_eligible),
        "max_batch_tasks": int(max_batch_tasks),
        "launch_delay_ns_by_link_class": launch_delay_ns_by_link_class,
        "fixed_latency_ns_by_link_class": fixed_latency_ns_by_link_class,
        "bandwidth_bytes_per_second_by_link_class": bandwidth_bytes_per_second_by_link_class,
    }
    return HardwareProfile(
        profile_id=str(profile_id),
        profile_digest=stable_digest(semantic, domain="HARDWARE_PROFILE"),
        profile_provenance=str(profile_provenance),
        performance_eligible=bool(performance_eligible),
        max_batch_tasks=int(max_batch_tasks),
        launch_delay_ns_by_link_class=launch_delay_ns_by_link_class,
        fixed_latency_ns_by_link_class=fixed_latency_ns_by_link_class,
        bandwidth_bytes_per_second_by_link_class=bandwidth_bytes_per_second_by_link_class,
    )


def make_control_plane_profile(
    *,
    profile_id: str,
    profile_provenance: str,
    performance_eligible: bool,
    fixed_latency_ns: int,
    bandwidth_bytes_per_second: int,
) -> ControlPlaneProfile:
    semantic = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "profile_id": str(profile_id),
        "profile_provenance": str(profile_provenance),
        "performance_eligible": bool(performance_eligible),
        "fixed_latency_ns": int(fixed_latency_ns),
        "bandwidth_bytes_per_second": int(bandwidth_bytes_per_second),
        "channel_count": 1,
        "fifo": True,
        "non_preemptive": True,
        "shares_data_nic": False,
    }
    return ControlPlaneProfile(
        profile_id=str(profile_id),
        profile_digest=stable_digest(semantic, domain="CONTROL_PLANE_PROFILE"),
        profile_provenance=str(profile_provenance),
        performance_eligible=bool(performance_eligible),
        fixed_latency_ns=int(fixed_latency_ns),
        bandwidth_bytes_per_second=int(bandwidth_bytes_per_second),
    )


def make_transport_snapshot(
    *,
    snapshot_at_ns: int,
    max_batch_tasks: int,
    busy_src_ranks: tuple[int, ...],
    busy_dst_ranks: tuple[int, ...],
    busy_nic_ids: tuple[str, ...],
    busy_lane_ids: tuple[str, ...],
    available_lane_ids_by_link_class: tuple[tuple[LinkClass, tuple[str, ...]], ...],
    hardware_profile_digest: str,
    topology_digest: str,
) -> TransportSnapshot:
    return TransportSnapshot(
        snapshot_at_ns=int(snapshot_at_ns),
        max_batch_tasks=int(max_batch_tasks),
        busy_src_ranks=tuple(sorted(int(v) for v in busy_src_ranks)),
        busy_dst_ranks=tuple(sorted(int(v) for v in busy_dst_ranks)),
        busy_nic_ids=tuple(sorted(str(v) for v in busy_nic_ids)),
        busy_lane_ids=tuple(sorted(str(v) for v in busy_lane_ids)),
        available_lane_ids_by_link_class=tuple(
            (link_class, tuple(sorted(str(v) for v in lane_ids)))
            for link_class, lane_ids in available_lane_ids_by_link_class
        ),
        hardware_profile_digest=str(hardware_profile_digest),
        topology_digest=str(topology_digest),
    )


def make_commit_receipt(
    *,
    batch: TransferBatch,
    commit_time_ns: int,
    resource_reservation_digest: str,
    transport_snapshot_digest: str,
) -> CommitReceipt:
    semantic = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "batch_digest": batch.batch_digest,
        "phase_key": batch.phase_key,
        "task_ids": batch.task_ids,
        "authority_stamp": batch.authority_stamp,
        "topology_digest": batch.topology_digest,
        "commit_time_ns": int(commit_time_ns),
        "resource_reservation_digest": str(resource_reservation_digest),
        "transport_snapshot_digest": str(transport_snapshot_digest),
    }
    return CommitReceipt(
        receipt_id=stable_digest(semantic, domain="COMMIT_RECEIPT"),
        batch_id=batch.batch_id,
        batch_digest=batch.batch_digest,
        phase_key=batch.phase_key,
        task_ids=batch.task_ids,
        authority_stamp=batch.authority_stamp,
        topology_digest=batch.topology_digest,
        commit_time_ns=int(commit_time_ns),
        resource_reservation_digest=str(resource_reservation_digest),
        transport_snapshot_digest=str(transport_snapshot_digest),
    )

def hardware_profile_digest(profile: HardwareProfile) -> str:
    return stable_digest(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "profile_id": profile.profile_id,
            "profile_provenance": profile.profile_provenance,
            "performance_eligible": profile.performance_eligible,
            "max_batch_tasks": profile.max_batch_tasks,
            "launch_delay_ns_by_link_class": profile.launch_delay_ns_by_link_class,
            "fixed_latency_ns_by_link_class": profile.fixed_latency_ns_by_link_class,
            "bandwidth_bytes_per_second_by_link_class": profile.bandwidth_bytes_per_second_by_link_class,
        },
        domain="HARDWARE_PROFILE",
    )


def control_plane_profile_digest(profile: ControlPlaneProfile) -> str:
    return stable_digest(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "profile_id": profile.profile_id,
            "profile_provenance": profile.profile_provenance,
            "performance_eligible": profile.performance_eligible,
            "fixed_latency_ns": profile.fixed_latency_ns,
            "bandwidth_bytes_per_second": profile.bandwidth_bytes_per_second,
            "channel_count": profile.channel_count,
            "fifo": profile.fifo,
            "non_preemptive": profile.non_preemptive,
            "shares_data_nic": profile.shares_data_nic,
        },
        domain="CONTROL_PLANE_PROFILE",
    )


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "ceil_transfer_time_ns",
    "control_plane_profile_digest",
    "descriptor_truth_digest",
    "hardware_profile_digest",
    "make_authority_stamp",
    "make_transport_snapshot",
    "make_hardware_profile",
    "make_control_plane_profile",
    "make_commit_receipt",
    "make_exact_dispatch_row_truth",
    "make_exact_row_descriptor",
    "make_network_topology",
    "make_row_broadcast_request",
    "make_task_resource_footprint",
    "make_transfer_batch",
]
