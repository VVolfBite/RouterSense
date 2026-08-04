from __future__ import annotations

import pytest

from rs_sim.backend import BackendContractError, CapacityConfigurationError
from rs_sim.backend.core.internal import ReceiverJobStatus

from tests.backend.conftest import Phase, make_system, make_task


def test_future_combine_does_not_reserve_before_payload_ready(phases):
    system = make_system(world_size=2, capacity=64, posting_fixed_ns=2)
    c0 = phases["c0"]
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[64, 0],
        created_at_ns=0,
    )
    task = make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=64, registered=5)
    system.backend.register_canonical_task_catalogue([task])

    record = system.receiver.task_record(task.task_id)
    assert record.requested_at_ns == 5
    assert record.eligible_at_ns is None
    assert system.receiver.current_memory(0)["reserved_bytes"] == 0
    assert system.kernel.next_time() == 0
    system.kernel.run_until(5)
    assert system.receiver.current_memory(0)["reserved_bytes"] == 0

    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=100)
    system.kernel.run_until(100)
    assert record.eligible_at_ns == 100
    assert record.receiver_start_ns == 100
    assert system.receiver.current_memory(0)["reserved_bytes"] == 64


def test_zero_edge_creates_no_job_permit_or_credit(phases):
    system = make_system(world_size=1, capacity=64)
    c0 = phases["c0"]
    system.backend.register_local_path_spec(
        combine_phase_key=c0,
        next_dispatch_phase_key=phases["d1"],
        rank_id=0,
        combine_release_to_router_ready_ns=0,
        router_and_pack_ns=0,
    )
    expectations = system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[0],
        created_at_ns=7,
    )
    assert expectations[0].zero_edge is True
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=10)
    assert system.receiver.current_memory(0)["staging_bytes"] == 0
    assert system.observer.times("RECEIVER_JOB_REQUESTED") == []
    assert system.observer.times("RECEIVE_PERMIT_GRANTED") == []


def test_task_ranges_must_cover_edge_exactly(phases):
    system = make_system(world_size=1, capacity=128)
    c0 = phases["c0"]
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[100],
        created_at_ns=0,
    )
    bad = [
        make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=40),
        make_task(phase=c0, src=0, dst=0, chunk=1, offset=50, size=50),
    ]
    with pytest.raises(BackendContractError, match="contiguous"):
        system.backend.register_canonical_task_catalogue(bad)


def test_capacity_must_hold_largest_canonical_task(phases):
    system = make_system(world_size=1, capacity=32)
    c0 = phases["c0"]
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[64],
        created_at_ns=0,
    )
    with pytest.raises(CapacityConfigurationError):
        system.backend.register_canonical_task_catalogue(
            [make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=64)]
        )


def test_single_fifo_hol_does_not_bypass_smaller_task(phases):
    system = make_system(
        world_size=1,
        capacity=100,
        posting_fixed_ns=0,
        drain_fixed_ns=9,
    )
    c0 = phases["c0"]
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[160],
        created_at_ns=0,
    )
    tasks = [
        make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=80),
        make_task(phase=c0, src=0, dst=0, chunk=1, offset=80, size=60),
        make_task(phase=c0, src=0, dst=0, chunk=2, offset=140, size=20),
    ]
    system.backend.register_canonical_task_catalogue(tasks)
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=0)

    # ceil(bytes / 1e9) = 1ns: task 0 posts at t=1.  The head task 1
    # needs 60 bytes while only 20 remain; task 2 would fit but cannot bypass.
    system.kernel.run_until(1)
    assert system.receiver.task_record(tasks[1].task_id).buffer_stall_started_ns == 1
    assert system.receiver.task_record(tasks[2].task_id).receiver_start_ns is None

    system.backend.on_transfer_completed(task_id=tasks[0].task_id, at_ns=2)
    system.kernel.run_until(12)
    assert system.receiver.task_record(tasks[1].task_id).receiver_start_ns == 12
    assert system.receiver.task_record(tasks[2].task_id).receiver_start_ns is None
    metrics = system.receiver.metrics_snapshot()
    assert metrics.receiver_buffer_stall_ns[0] == 11
    # Task 1 waited one nanosecond behind task 0's posting service and eleven
    # nanoseconds for capacity.  The two intervals are mutually exclusive.
    assert metrics.receiver_posting_queue_wait_ns[0] == 1


def test_quarter_capacity_progresses_when_capacity_equals_max_task(phases):
    # Reference inbound=256, 0.25x=>64, and max canonical task=64.
    system = make_system(world_size=1, capacity=64, drain_fixed_ns=1)
    c0 = phases["c0"]
    system.backend.register_local_path_spec(
        combine_phase_key=c0,
        next_dispatch_phase_key=phases["d1"],
        rank_id=0,
        combine_release_to_router_ready_ns=1,
        router_and_pack_ns=1,
    )
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[256],
        created_at_ns=0,
    )
    tasks = [
        make_task(phase=c0, src=0, dst=0, chunk=i, offset=64 * i, size=64)
        for i in range(4)
    ]
    system.backend.register_canonical_task_catalogue(tasks)
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=0)

    for task in tasks:
        record = system.receiver.task_record(task.task_id)
        while record.status != ReceiverJobStatus.POSTED:
            next_time = system.kernel.next_time()
            assert next_time is not None
            system.kernel.run_until(next_time)
        posted = record.receive_posted_at_ns
        assert posted is not None
        system.backend.on_transfer_completed(task_id=task.task_id, at_ns=posted + 1)
        system.kernel.run_until(posted + 3)

    system.kernel.run_until()
    assert all(
        system.receiver.task_record(task.task_id).status == ReceiverJobStatus.ASSEMBLED
        for task in tasks
    )
    assert system.observer.times("LOCAL_PATH_COMPLETE")


def test_same_time_eligibility_uses_semantic_fifo_not_callback_order(phases):
    system = make_system(world_size=2, capacity=128, posting_fixed_ns=5)
    c0 = phases["c0"]
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[64, 64],
        created_at_ns=0,
    )
    src0 = make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=64)
    src1 = make_task(phase=c0, src=1, dst=0, chunk=0, offset=0, size=64)
    system.backend.register_canonical_task_catalogue([src1, src0])

    # Intentionally publish in reverse source order at the same timestamp.
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=1, at_ns=10)
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=10)
    system.kernel.run_until(10)

    assert system.receiver.task_record(src0.task_id).receiver_start_ns == 10
    assert system.receiver.task_record(src1.task_id).receiver_start_ns is None


def test_same_time_transfer_completions_use_drain_fifo_not_callback_order(phases):
    system = make_system(
        world_size=1,
        capacity=128,
        posting_fixed_ns=0,
        drain_fixed_ns=5,
    )
    c0 = phases["c0"]
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[128],
        created_at_ns=0,
    )
    first = make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=64)
    second = make_task(phase=c0, src=0, dst=0, chunk=1, offset=64, size=64)
    system.backend.register_canonical_task_catalogue([second, first])
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=0)
    system.kernel.run_until(2)
    assert system.receiver.task_record(first.task_id).status == ReceiverJobStatus.POSTED
    assert system.receiver.task_record(second.task_id).status == ReceiverJobStatus.POSTED

    # Apply network completions in reverse order at one timestamp. The drain line
    # must still use its semantic FIFO key, not callback arrival order.
    system.backend.on_transfer_completed(task_id=second.task_id, at_ns=10)
    system.backend.on_transfer_completed(task_id=first.task_id, at_ns=10)
    system.kernel.run_until(10)

    assert system.receiver.task_record(first.task_id).drain_start_ns == 10
    assert system.receiver.task_record(second.task_id).drain_start_ns is None

    system.kernel.run_until(16)
    metrics = system.receiver.metrics_snapshot()
    # A 64-byte drain costs 5 fixed + 1 serialization nanosecond.  At t=16
    # the first service has finished and the second has waited exactly six ns.
    assert metrics.receiver_drain_queue_wait_ns[0] == 6
    assert metrics.receiver_drain_service_ns[0] == 6


def test_network_completion_cannot_precede_receive_permit(phases):
    system = make_system(world_size=1, capacity=64, posting_fixed_ns=5)
    c0 = phases["c0"]
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[64],
        created_at_ns=0,
    )
    task = make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=64)
    system.backend.register_canonical_task_catalogue([task])
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=0)
    system.kernel.run_until(6)
    assert system.receiver.task_record(task.task_id).status == ReceiverJobStatus.POSTED

    with pytest.raises(BackendContractError, match="cannot precede"):
        system.backend.on_transfer_completed(task_id=task.task_id, at_ns=5)


def test_source_payload_rank_must_be_inside_world(phases):
    system = make_system(world_size=1, capacity=64)
    with pytest.raises(BackendContractError, match="outside world_size"):
        system.backend.on_source_payload_ready(
            phase_key=phases["c0"], src_rank=1, at_ns=0
        )
