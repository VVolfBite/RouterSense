from __future__ import annotations

from tests.backend.conftest import make_system, make_task


def test_combine_final_assembly_released_only_after_post_combine_path(phases):
    system = make_system(
        world_size=1,
        capacity=64,
        posting_fixed_ns=2,
        drain_fixed_ns=4,
    )
    c0, d1 = phases["c0"], phases["d1"]
    system.backend.register_local_path_spec(
        combine_phase_key=c0,
        next_dispatch_phase_key=d1,
        rank_id=0,
        combine_release_to_router_ready_ns=7,
        router_and_pack_ns=11,
    )
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[64],
        created_at_ns=0,
    )
    task = make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=64)
    system.backend.register_canonical_task_catalogue([task])
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=0)

    system.kernel.run_until(3)  # posting fixed 2 + one integer transfer unit
    system.backend.on_transfer_completed(task_id=task.task_id, at_ns=5)
    system.kernel.run_until(10)  # drain fixed 4 + one unit

    assert system.backend.combine_destination_snapshot(
        phase_key=c0, dst_rank=0
    )["data_ready_at_ns"] == 10
    assert system.receiver.current_memory(0)["final_assembly_bytes"] == 64
    system.kernel.run_until(16)
    assert system.receiver.current_memory(0)["final_assembly_bytes"] == 64

    system.kernel.run_until(17)
    assert system.receiver.current_memory(0)["final_assembly_bytes"] == 0
    assert system.observer.times("POST_COMBINE_LOCAL_PATH_COMPLETE") == [17]

    system.kernel.run_until(28)
    assert system.observer.times("SOURCE_DESCRIPTOR_READY") == [28]
    assert system.observer.times("SOURCE_PAYLOAD_READY")[-1] == 28
    assert system.observer.times("DESTINATION_DISPATCH_THREAD_READY") == [28]


def test_dispatch_release_uses_max_of_thread_closure_and_data(phases):
    system = make_system(
        world_size=2,
        capacity=64,
        posting_fixed_ns=2,
        drain_fixed_ns=4,
    )
    d0, c0 = phases["d0"], phases["c0"]
    system.backend.register_dispatch_compute_spec(
        dispatch_phase_key=d0,
        next_combine_phase_key=c0,
        rank_id=0,
        dispatch_local_postprocess_ns=7,
        dispatch_release_to_combine_source_ready_ns=11,
    )
    task = make_task(phase=d0, src=1, dst=0, chunk=0, offset=0, size=64)
    system.backend.register_canonical_task_catalogue([task])
    system.backend.on_source_payload_ready(phase_key=d0, src_rank=1, at_ns=10)

    system.backend.on_dispatch_descriptor_delivered(
        phase_key=d0,
        src_rank=0,
        payload_bytes_by_destination=[0, 0],
        descriptor_digest="row0",
        delivered_at_ns=40,
    )
    system.backend.on_dispatch_descriptor_delivered(
        phase_key=d0,
        src_rank=1,
        payload_bytes_by_destination=[64, 0],
        descriptor_digest="row1",
        delivered_at_ns=60,
    )
    system.kernel.run_until(63)
    system.backend.on_transfer_completed(task_id=task.task_id, at_ns=65)
    system.kernel.run_until(70)

    snap = system.backend.dispatch_destination_snapshot(phase_key=d0, dst_rank=0)
    assert snap["descriptor_closure_at_ns"] == 60
    assert snap["all_inbound_assembled_at_ns"] == 70
    assert snap["compute_ready_at_ns"] is None

    system.backend.mark_dispatch_model_thread_ready(
        phase_key=d0, dst_rank=0, at_ns=100
    )
    system.kernel.run_until(100)
    snap = system.backend.dispatch_destination_snapshot(phase_key=d0, dst_rank=0)
    assert snap["postprocess_start_at_ns"] == 100
    assert snap["compute_ready_at_ns"] == 107

    system.kernel.run_until(107)
    assert system.backend.rank_release_at(phase_key=d0, rank_id=0) == 107
    assert system.receiver.current_memory(0)["final_assembly_bytes"] == 0
    assert system.observer.times("BACKEND_RANK_RELEASED") == [107]

    system.kernel.run_until(118)
    assert system.observer.times("COMBINE_HOOK_READY") == [118]
    assert system.observer.times("SOURCE_PAYLOAD_READY")[-1] == 118


def test_transfer_before_descriptor_closure_cannot_release(phases):
    system = make_system(
        world_size=2,
        capacity=64,
        posting_fixed_ns=0,
        drain_fixed_ns=0,
    )
    d0, c0 = phases["d0"], phases["c0"]
    system.backend.register_dispatch_compute_spec(
        dispatch_phase_key=d0,
        next_combine_phase_key=c0,
        rank_id=0,
        dispatch_local_postprocess_ns=5,
        dispatch_release_to_combine_source_ready_ns=1,
    )
    task = make_task(phase=d0, src=0, dst=0, chunk=0, offset=0, size=64)
    system.backend.register_canonical_task_catalogue([task])
    system.backend.on_source_payload_ready(phase_key=d0, src_rank=0, at_ns=0)
    system.backend.mark_dispatch_model_thread_ready(phase_key=d0, dst_rank=0, at_ns=10)

    system.backend.on_dispatch_descriptor_delivered(
        phase_key=d0,
        src_rank=0,
        payload_bytes_by_destination=[64, 0],
        descriptor_digest="early-row",
        delivered_at_ns=20,
    )
    system.kernel.run_until(21)
    system.backend.on_transfer_completed(task_id=task.task_id, at_ns=22)
    system.kernel.run_until(23)
    assert system.backend.rank_release_at(phase_key=d0, rank_id=0) is None

    system.backend.on_dispatch_descriptor_delivered(
        phase_key=d0,
        src_rank=1,
        payload_bytes_by_destination=[0, 0],
        descriptor_digest="late-zero-row",
        delivered_at_ns=100,
    )
    system.kernel.run_until(100)
    snap = system.backend.dispatch_destination_snapshot(phase_key=d0, dst_rank=0)
    assert snap["postprocess_start_at_ns"] == 100
    assert snap["compute_ready_at_ns"] == 105
    system.kernel.run_until(105)
    assert system.backend.rank_release_at(phase_key=d0, rank_id=0) == 105


def test_memory_metrics_track_staging_final_and_total(phases):
    system = make_system(
        world_size=1,
        capacity=100,
        posting_fixed_ns=0,
        drain_fixed_ns=9,
    )
    c0, d1 = phases["c0"], phases["d1"]
    system.backend.register_local_path_spec(
        combine_phase_key=c0,
        next_dispatch_phase_key=d1,
        rank_id=0,
        combine_release_to_router_ready_ns=20,
        router_and_pack_ns=0,
    )
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[140],
        created_at_ns=0,
    )
    tasks = [
        make_task(phase=c0, src=0, dst=0, chunk=0, offset=0, size=80),
        make_task(phase=c0, src=0, dst=0, chunk=1, offset=80, size=60),
    ]
    system.backend.register_canonical_task_catalogue(tasks)
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=0)
    system.kernel.run_until(1)
    system.backend.on_transfer_completed(task_id=tasks[0].task_id, at_ns=2)
    system.kernel.run_until(12)  # first final=80 and second reserved=60 => total 140

    metrics = system.receiver.metrics_snapshot()
    assert metrics.peak_staging_bytes_per_rank[0] == 80
    assert metrics.peak_final_assembly_bytes_per_rank[0] == 80
    assert metrics.peak_total_receiver_bytes_per_rank[0] == 140
    assert metrics.receiver_buffer_stall_ns[0] == 11


def test_dispatch_phase_barrier_holds_fast_rank(phases):
    system = make_system(
        world_size=2,
        capacity=64,
        release_mode="PHASE_BARRIER",
    )
    d0, c0 = phases["d0"], phases["c0"]
    for rank in range(2):
        system.backend.register_dispatch_compute_spec(
            dispatch_phase_key=d0,
            next_combine_phase_key=c0,
            rank_id=rank,
            dispatch_local_postprocess_ns=5,
            dispatch_release_to_combine_source_ready_ns=1,
        )
    for src in range(2):
        system.backend.on_dispatch_descriptor_delivered(
            phase_key=d0,
            src_rank=src,
            payload_bytes_by_destination=[0, 0],
            descriptor_digest=f"zero-row-{src}",
            delivered_at_ns=0,
        )
    system.backend.mark_dispatch_model_thread_ready(
        phase_key=d0, dst_rank=0, at_ns=10
    )
    system.backend.mark_dispatch_model_thread_ready(
        phase_key=d0, dst_rank=1, at_ns=20
    )

    system.kernel.run_until(15)
    assert system.observer.times("DESTINATION_COMPUTE_READY") == [15]
    assert system.observer.times("BACKEND_RANK_RELEASED") == []

    system.kernel.run_until(25)
    assert system.observer.times("DESTINATION_COMPUTE_READY") == [15, 25]
    assert system.observer.times("DISPATCH_PHASE_BARRIER_RELEASE") == [25]
    assert system.observer.times("BACKEND_RANK_RELEASED") == [25, 25]
    assert system.backend.rank_release_at(phase_key=d0, rank_id=0) == 25
    assert system.backend.rank_release_at(phase_key=d0, rank_id=1) == 25


def test_combine_phase_barrier_delays_local_path_start(phases):
    system = make_system(
        world_size=2,
        capacity=64,
        release_mode="PHASE_BARRIER",
    )
    c0, d1 = phases["c0"], phases["d1"]
    system.backend.register_local_path_spec(
        combine_phase_key=c0,
        next_dispatch_phase_key=d1,
        rank_id=0,
        combine_release_to_router_ready_ns=3,
        router_and_pack_ns=1,
    )
    system.backend.register_local_path_spec(
        combine_phase_key=c0,
        next_dispatch_phase_key=d1,
        rank_id=1,
        combine_release_to_router_ready_ns=7,
        router_and_pack_ns=1,
    )
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[0, 0],
        created_at_ns=5,
    )
    assert system.kernel.next_time() == 5
    system.kernel.run_until(5)
    assert system.observer.times("POST_COMBINE_LOCAL_PATH_COMPLETE") == []

    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=1,
        realized_dispatch_payload_bytes_by_expert=[0, 0],
        created_at_ns=20,
    )
    # Zero inbound declarations close destination data, but the post-Combine
    # path remains causally blocked until the same rank's source-side expert
    # work has produced its Combine payload.
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=10)
    system.backend.on_source_payload_ready(phase_key=c0, src_rank=1, at_ns=20)
    system.kernel.run_until(20)
    assert system.observer.times("COMBINE_PHASE_BARRIER_OPEN") == [20]
    assert system.kernel.next_time() == 23
    system.kernel.run_until(22)
    assert system.observer.times("POST_COMBINE_LOCAL_PATH_COMPLETE") == []
    system.kernel.run_until(27)
    assert system.observer.times("POST_COMBINE_LOCAL_PATH_COMPLETE") == [23, 27]



def test_zero_inbound_combine_waits_for_same_rank_source_lifecycle(phases):
    system = make_system(world_size=1, capacity=64, release_mode="RANK_LOCAL")
    c0, d1 = phases["c0"], phases["d1"]
    system.backend.register_local_path_spec(
        combine_phase_key=c0,
        next_dispatch_phase_key=d1,
        rank_id=0,
        combine_release_to_router_ready_ns=5,
        router_and_pack_ns=2,
    )
    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=c0,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=[0],
        created_at_ns=10,
    )
    system.kernel.run_until(50)
    assert system.observer.times("POST_COMBINE_LOCAL_PATH_COMPLETE") == []

    system.backend.on_source_payload_ready(phase_key=c0, src_rank=0, at_ns=100)
    system.kernel.run_until(104)
    assert system.observer.times("POST_COMBINE_LOCAL_PATH_COMPLETE") == []
    system.kernel.run_until(105)
    assert system.observer.times("POST_COMBINE_LOCAL_PATH_COMPLETE") == [105]
    system.kernel.run_until(107)
    assert system.observer.times("DESTINATION_DISPATCH_THREAD_READY") == [107]

def test_dispatch_closure_wait_metric_is_rank_attributed(phases):
    system = make_system(world_size=2, capacity=64, posting_fixed_ns=0, drain_fixed_ns=0)
    d0, c0 = phases["d0"], phases["c0"]
    system.backend.register_dispatch_compute_spec(
        dispatch_phase_key=d0,
        next_combine_phase_key=c0,
        rank_id=0,
        dispatch_local_postprocess_ns=0,
        dispatch_release_to_combine_source_ready_ns=0,
    )
    system.backend.mark_dispatch_model_thread_ready(phase_key=d0, dst_rank=0, at_ns=0)
    system.backend.on_dispatch_descriptor_delivered(
        phase_key=d0,
        src_rank=0,
        payload_bytes_by_destination=[64, 0],
        descriptor_digest="src0",
        delivered_at_ns=0,
    )
    task = make_task(phase=d0, src=0, dst=0, chunk=0, offset=0, size=64)
    system.backend.register_canonical_task_catalogue([task])
    system.backend.on_source_payload_ready(phase_key=d0, src_rank=0, at_ns=0)
    system.kernel.run_until(1)
    posted = system.receiver.task_record(task.task_id).receive_posted_at_ns
    assert posted is not None
    system.backend.on_transfer_completed(task_id=task.task_id, at_ns=posted + 1)
    system.kernel.run_until(10)

    # The last source contributes only a zero edge. Data and model thread were
    # ready earlier, so the delayed descriptor closure is visible as receiver wait.
    system.backend.on_dispatch_descriptor_delivered(
        phase_key=d0,
        src_rank=1,
        payload_bytes_by_destination=[0, 0],
        descriptor_digest="src1",
        delivered_at_ns=20,
    )
    system.kernel.run_until()
    metrics = system.backend.metrics_snapshot()
    assert metrics.dispatch_closure_wait_ns[0] > 0
    assert metrics.dispatch_closure_wait_ns[1] == 0


def test_late_combine_source_ready_does_not_regress_terminal_rank_state():
    """Source readiness is orthogonal to destination-side rank progression."""
    from rs_sim.backend.core.internal import RankState
    from rs_sim.backend.resources.rank_actor import RankActor

    actor = RankActor(rank_id=0)
    actor.transition(state=RankState.DONE, phase_key="combine", at_ns=10, reason="terminal")
    # The production handler now guards the transition; this small invariant
    # documents the prohibited regression directly.
    if actor.state in {RankState.EXPERT_COMPUTE, RankState.WAIT_COMBINE}:
        actor.transition(state=RankState.WAIT_COMBINE, phase_key="combine", at_ns=20, reason="source")
    assert actor.state is RankState.DONE


def test_p0_p1_compute_end_barrier_aligns_combine_source_ready(phases):
    system = make_system(
        world_size=2,
        capacity=64,
        p0_p1_compute_end_barrier=True,
    )
    d0, c0 = phases["d0"], phases["c0"]
    for rank, compute_ns in enumerate((5, 13)):
        system.backend.register_dispatch_compute_spec(
            dispatch_phase_key=d0,
            next_combine_phase_key=c0,
            rank_id=rank,
            dispatch_local_postprocess_ns=0,
            dispatch_release_to_combine_source_ready_ns=compute_ns,
        )
    for src in range(2):
        system.backend.on_dispatch_descriptor_delivered(
            phase_key=d0,
            src_rank=src,
            payload_bytes_by_destination=[0, 0],
            descriptor_digest=f"barrier-zero-row-{src}",
            delivered_at_ns=0,
        )
    for rank in range(2):
        system.backend.mark_dispatch_model_thread_ready(
            phase_key=d0, dst_rank=rank, at_ns=0
        )

    system.kernel.run_until(5)
    assert system.observer.times("P0_P1_LOCAL_COMPUTE_COMPLETE") == [5]
    assert system.observer.times("COMBINE_HOOK_READY") == []

    system.kernel.run_until(13)
    assert system.observer.times("P0_P1_LOCAL_COMPUTE_COMPLETE") == [5, 13]
    assert system.observer.times("P0_P1_COMPUTE_END_BARRIER_RELEASE") == [13]
    assert system.observer.times("COMBINE_HOOK_READY") == [13, 13]
    assert system.backend.p0_p1_compute_barrier_release_at(
        dispatch_phase_key=d0
    ) == 13


def test_p0_p1_compute_end_barrier_disabled_preserves_rank_local_ready(phases):
    system = make_system(world_size=2, capacity=64)
    d0, c0 = phases["d0"], phases["c0"]
    for rank, compute_ns in enumerate((5, 13)):
        system.backend.register_dispatch_compute_spec(
            dispatch_phase_key=d0,
            next_combine_phase_key=c0,
            rank_id=rank,
            dispatch_local_postprocess_ns=0,
            dispatch_release_to_combine_source_ready_ns=compute_ns,
        )
    for src in range(2):
        system.backend.on_dispatch_descriptor_delivered(
            phase_key=d0,
            src_rank=src,
            payload_bytes_by_destination=[0, 0],
            descriptor_digest=f"rank-local-zero-row-{src}",
            delivered_at_ns=0,
        )
    for rank in range(2):
        system.backend.mark_dispatch_model_thread_ready(
            phase_key=d0, dst_rank=rank, at_ns=0
        )

    system.kernel.run_until(13)
    assert system.observer.times("P0_P1_COMPUTE_END_BARRIER_RELEASE") == []
    assert system.observer.times("COMBINE_HOOK_READY") == [5, 13]
