from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rs_sim import (
    CanonicalTransferTask,
    EdgeKey,
    ExpectationOrigin,
    PhaseExecutionRecord,
    PhaseKey,
    PhaseKind,
    PlanStatus,
    PlanVersion,
    ReceiveExpectation,
    ReceivePermit,
    StableSerializationError,
    WindowKey,
    stable_digest,
    stable_json_dumps,
)


def keys():
    window = WindowKey("run-a", "sample-0", 0)
    phase = PhaseKey("run-a", "sample-0", 3, PhaseKind.DISPATCH)
    edge = EdgeKey(phase, 0, 1)
    return window, phase, edge


def test_shared_objects_are_frozen_and_digest_stable() -> None:
    window, phase, edge = keys()
    expectation = ReceiveExpectation(
        edge_key=edge,
        phase_key=phase,
        src_rank=0,
        dst_rank=1,
        total_expected_payload_bytes=4096,
        expectation_digest="exp-digest",
        origin=ExpectationOrigin.DISPATCH_DESCRIPTOR,
        created_at_ns=10,
        zero_edge=False,
        descriptor_digest_or_none="row-digest",
    )
    task = CanonicalTransferTask(
        task_id="task-0",
        edge_key=edge,
        phase_key=phase,
        src_rank=0,
        dst_rank=1,
        chunk_index=0,
        byte_offset=0,
        payload_bytes=4096,
        expectation_digest=expectation.expectation_digest,
        taskization_digest="taskization-digest",
        registered_at_ns=12,
    )
    permit = ReceivePermit(
        permit_id="permit-0",
        task_id=task.task_id,
        edge_key=edge,
        chunk_index=0,
        byte_offset=0,
        task_bytes=4096,
        credit_reservation_id="credit-0",
        expectation_digest=expectation.expectation_digest,
        descriptor_digest_or_none="row-digest",
        posted_at_ns=15,
    )
    record = PhaseExecutionRecord(
        phase_key=phase,
        canonical_task_ids=("task-1", "task-0"),
        task_catalogue_digest="catalogue-digest",
        active_plan_id="plan-0",
        phase_plan_epoch=2,
        committed_task_ids=("task-0",),
        running_task_ids=(),
        completed_task_ids=(),
        registered_window_keys=(window,),
    )
    plan = PlanVersion(
        plan_id="plan-0",
        window_key=window,
        version=0,
        status=PlanStatus.ACTIVE,
        supersedes_plan_ids=(),
        commit_index=0,
        committed_task_ids=("task-0",),
        remaining_task_ids=("task-1",),
        created_at_ns=20,
        activated_at_ns=21,
        completed_at_ns=None,
        plan_digest="plan-digest",
    )

    fixture = (expectation, task, permit, record, plan)
    assert stable_digest(fixture) == stable_digest(fixture)
    assert record.canonical_task_ids == ("task-1", "task-0")
    with pytest.raises(FrozenInstanceError):
        task.payload_bytes = 1  # type: ignore[misc]


def test_zero_edge_and_dispatch_combine_digest_rules() -> None:
    _, dispatch, dispatch_edge = keys()
    with pytest.raises(ValueError, match="zero_edge"):
        ReceiveExpectation(
            edge_key=dispatch_edge,
            phase_key=dispatch,
            src_rank=0,
            dst_rank=1,
            total_expected_payload_bytes=0,
            expectation_digest="exp",
            origin=ExpectationOrigin.DISPATCH_DESCRIPTOR,
            created_at_ns=0,
            zero_edge=False,
            descriptor_digest_or_none="row",
        )

    combine = PhaseKey("run-a", "sample-0", 3, PhaseKind.COMBINE)
    combine_edge = EdgeKey(combine, 1, 0)
    with pytest.raises(ValueError, match="must be None"):
        ReceiveExpectation(
            edge_key=combine_edge,
            phase_key=combine,
            src_rank=1,
            dst_rank=0,
            total_expected_payload_bytes=32,
            expectation_digest="exp",
            origin=ExpectationOrigin.COMBINE_REALIZED,
            created_at_ns=0,
            zero_edge=False,
            descriptor_digest_or_none="illegal",
        )


def test_stable_codec_rejects_float_mutable_and_unordered_values() -> None:
    with pytest.raises(StableSerializationError):
        stable_json_dumps(1.5)
    with pytest.raises(StableSerializationError):
        stable_json_dumps([1, 2])
    with pytest.raises(StableSerializationError):
        stable_json_dumps({1, 2})


def test_transport_public_contract_constructs_without_float_time():
    from rs_sim import (
        AuthorityStamp,
        CommitReceipt,
        LinkClass,
        PhaseKey,
        PhaseKind,
        TransportSnapshot,
    )

    phase = PhaseKey("run", "sample", 0, PhaseKind.DISPATCH)
    snapshot = TransportSnapshot(
        snapshot_at_ns=7,
        max_batch_tasks=2,
        busy_src_ranks=(),
        busy_dst_ranks=(),
        busy_nic_ids=(),
        busy_lane_ids=(),
        available_lane_ids_by_link_class=((LinkClass.INTER_NODE, ("lane0",)),),
        hardware_profile_digest="profile",
        topology_digest="topology",
    )
    stamp = AuthorityStamp(
        phase_token="phase-token",
        plan_id="plan",
        phase_plan_epoch=1,
        authority_digest="authority-digest",
    )
    receipt = CommitReceipt(
        receipt_id="receipt",
        batch_id="batch",
        batch_digest="batch-digest",
        phase_key=phase,
        task_ids=("task",),
        authority_stamp=stamp,
        topology_digest="topology",
        commit_time_ns=7,
        resource_reservation_digest="resource",
        transport_snapshot_digest="snapshot",
    )
    assert snapshot.snapshot_at_ns == receipt.commit_time_ns
