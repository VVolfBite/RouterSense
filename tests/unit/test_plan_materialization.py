from __future__ import annotations

from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan
from rs.core.contracts.execution import ActualPhaseContext, MaterializedPlan, PayloadSpec, TransferSlice, ExecutionBatch
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.materialization import CommonPlanMaterializer, CommonPlanValidator
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def _build_window_plan() -> WindowPlan:
    return WindowPlan(
        planner_id="future:p012:joint:global:rscf",
        planner_family="joint",
        request_digest="0->1",
        waves=(
            PlanWave(
                wave_id=0,
                flows=(
                    PlannedFlow(
                        flow_id="p0_0_1",
                        phase="p0_dispatch",
                        src_rank=0,
                        dst_rank=1,
                        row_count=48,
                        release_state="ready",
                        executable=True,
                    ),
                    PlannedFlow(
                        flow_id="p0_1_0",
                        phase="p0_dispatch",
                        src_rank=1,
                        dst_rank=0,
                        row_count=32,
                        release_state="ready",
                        executable=True,
                    ),
                ),
                estimated_duration=48.0,
            ),
        ),
        metadata={"source_layer_id": "0", "target_layer_id": "1"},
    )


def _build_materialized_plan():
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 48), (32, 0)), p2_hint_mode="deterministic_stub")
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=(0, 1), root_rank=0))
    published = publisher.build(
        publication_slot={
            "run_id": "run",
            "forward_generation": 0,
            "microbatch_id": "mb",
            "source_layer_id": "0",
            "target_layer_id": "1",
            "planning_slot": "0->1",
        },
        window_plan=_build_window_plan(),
    )
    actual_context = ActualPhaseContext(
        layer_id="0",
        phase="P0",
        world_size=2,
        rank_space="global",
        layout_digest=str(contexts[0].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[0].to_dict()},
    )
    materialized = CommonPlanMaterializer().materialize(published, actual_context)
    return contexts[0], actual_context, materialized


def test_materializer_builds_materialized_plan_and_validator_accepts() -> None:
    _, actual_context, materialized = _build_materialized_plan()
    validation = CommonPlanValidator().validate(materialized, actual_context)
    assert validation.valid is True
    assert materialized.materialized_plan_digest
    assert materialized.batches


def test_validator_rejects_layout_digest_mismatch() -> None:
    _, actual_context, materialized = _build_materialized_plan()
    broken = ActualPhaseContext(
        layer_id=actual_context.layer_id,
        phase=actual_context.phase,
        world_size=actual_context.world_size,
        rank_space=actual_context.rank_space,
        layout_digest="wrong-layout",
        metadata=actual_context.metadata,
    )
    validation = CommonPlanValidator().validate(materialized, broken)
    assert validation.valid is False
    assert validation.reason == "layout_digest_mismatch"


def test_validator_rejects_payload_role_mismatch() -> None:
    _, actual_context, materialized = _build_materialized_plan()
    batch = materialized.batches[0]
    broken_slice = batch.slices[0]
    broken_batch = ExecutionBatch(
        batch_id=batch.batch_id,
        wave_id=batch.wave_id,
        phase=batch.phase,
        slices=(
                TransferSlice(
                    flow_id=broken_slice.flow_id,
                    task_id=broken_slice.task_id,
                    payload_role="unexpected_role",
                    src_group_rank=broken_slice.src_group_rank,
                    dst_group_rank=broken_slice.dst_group_rank,
                    src_global_rank=broken_slice.src_global_rank,
                    dst_global_rank=broken_slice.dst_global_rank,
                    row_count=broken_slice.row_count,
                    send_offset_rows=broken_slice.send_offset_rows,
                    recv_offset_rows=broken_slice.recv_offset_rows,
                dependency_ids=broken_slice.dependency_ids,
            ),
        )
        + batch.slices[1:],
        collective_required=batch.collective_required,
        metadata=batch.metadata,
    )
    broken_plan = MaterializedPlan(
        publication_slot=materialized.publication_slot,
        local_global_rank=materialized.local_global_rank,
        local_group_rank=materialized.local_group_rank,
        phase=materialized.phase,
        payload_specs=materialized.payload_specs,
        batches=(broken_batch,) + materialized.batches[1:],
        rank_map=materialized.rank_map,
        expected_outgoing_rows=materialized.expected_outgoing_rows,
        expected_incoming_rows=materialized.expected_incoming_rows,
        logical_plan_digest=materialized.logical_plan_digest,
        published_plan_digest=materialized.published_plan_digest,
        layout_digest=materialized.layout_digest,
        materialized_plan_digest="pending",
        metadata=materialized.metadata,
    )
    broken_plan = MaterializedPlan(
        publication_slot=broken_plan.publication_slot,
        local_global_rank=broken_plan.local_global_rank,
        local_group_rank=broken_plan.local_group_rank,
        phase=broken_plan.phase,
        payload_specs=broken_plan.payload_specs,
        batches=broken_plan.batches,
        rank_map=broken_plan.rank_map,
        expected_outgoing_rows=broken_plan.expected_outgoing_rows,
        expected_incoming_rows=broken_plan.expected_incoming_rows,
        logical_plan_digest=broken_plan.logical_plan_digest,
        published_plan_digest=broken_plan.published_plan_digest,
        layout_digest=broken_plan.layout_digest,
        materialized_plan_digest=broken_plan.recompute_materialized_plan_digest(),
        metadata=broken_plan.metadata,
    )
    validation = CommonPlanValidator().validate(broken_plan, actual_context)
    assert validation.valid is False


def test_validator_rejects_missing_first_slice_with_gap() -> None:
    _, actual_context, materialized = _build_materialized_plan()
    first_batch = materialized.batches[0]
    broken_batch = ExecutionBatch(
        batch_id=first_batch.batch_id,
        wave_id=first_batch.wave_id,
        phase=first_batch.phase,
        slices=first_batch.slices[1:],
        collective_required=first_batch.collective_required,
        metadata=first_batch.metadata,
    )
    broken_plan = MaterializedPlan(
        publication_slot=materialized.publication_slot,
        local_global_rank=materialized.local_global_rank,
        local_group_rank=materialized.local_group_rank,
        phase=materialized.phase,
        payload_specs=materialized.payload_specs,
        batches=(broken_batch,),
        rank_map=materialized.rank_map,
        expected_outgoing_rows=materialized.expected_outgoing_rows,
        expected_incoming_rows=materialized.expected_incoming_rows,
        logical_plan_digest=materialized.logical_plan_digest,
        published_plan_digest=materialized.published_plan_digest,
        layout_digest=materialized.layout_digest,
        materialized_plan_digest="pending",
        metadata=materialized.metadata,
    )
    broken_plan = MaterializedPlan(
        publication_slot=broken_plan.publication_slot,
        local_global_rank=broken_plan.local_global_rank,
        local_group_rank=broken_plan.local_group_rank,
        phase=broken_plan.phase,
        payload_specs=broken_plan.payload_specs,
        batches=broken_plan.batches,
        rank_map=broken_plan.rank_map,
        expected_outgoing_rows=broken_plan.expected_outgoing_rows,
        expected_incoming_rows=broken_plan.expected_incoming_rows,
        logical_plan_digest=broken_plan.logical_plan_digest,
        published_plan_digest=broken_plan.published_plan_digest,
        layout_digest=broken_plan.layout_digest,
        materialized_plan_digest=broken_plan.recompute_materialized_plan_digest(),
        metadata=broken_plan.metadata,
    )
    validation = CommonPlanValidator().validate(broken_plan, actual_context)
    assert validation.valid is False
    assert validation.reason in {"send_offset_gap", "recv_offset_gap"}


def test_validator_rejects_materialized_digest_mismatch() -> None:
    _, actual_context, materialized = _build_materialized_plan()
    broken_plan = MaterializedPlan(
        publication_slot=materialized.publication_slot,
        local_global_rank=materialized.local_global_rank,
        local_group_rank=materialized.local_group_rank,
        phase=materialized.phase,
        payload_specs=materialized.payload_specs,
        batches=materialized.batches,
        rank_map=materialized.rank_map,
        expected_outgoing_rows=materialized.expected_outgoing_rows,
        expected_incoming_rows=materialized.expected_incoming_rows,
        logical_plan_digest=materialized.logical_plan_digest,
        published_plan_digest=materialized.published_plan_digest,
        layout_digest=materialized.layout_digest,
        materialized_plan_digest="tampered",
        metadata=materialized.metadata,
    )
    validation = CommonPlanValidator().validate(broken_plan, actual_context)
    assert validation.valid is False
    assert validation.reason == "materialized_digest_mismatch"


def test_validator_rejects_duplicate_slice_overlap() -> None:
    _, actual_context, materialized = _build_materialized_plan()
    batch = materialized.batches[0]
    duplicated = batch.slices[0]
    broken_batch = ExecutionBatch(
        batch_id=batch.batch_id,
        wave_id=batch.wave_id,
        phase=batch.phase,
        slices=batch.slices + (
            TransferSlice(
                task_id="overlap-extra-task",
                flow_id=f"{duplicated.flow_id}:overlap",
                payload_role=duplicated.payload_role,
                src_group_rank=duplicated.src_group_rank,
                dst_group_rank=duplicated.dst_group_rank,
                src_global_rank=duplicated.src_global_rank,
                dst_global_rank=duplicated.dst_global_rank,
                row_count=duplicated.row_count,
                send_offset_rows=duplicated.send_offset_rows,
                recv_offset_rows=duplicated.recv_offset_rows,
                dependency_ids=duplicated.dependency_ids,
            ),
        ),
        collective_required=batch.collective_required,
        metadata=batch.metadata,
    )
    broken_plan = MaterializedPlan(
        publication_slot=materialized.publication_slot,
        local_global_rank=materialized.local_global_rank,
        local_group_rank=materialized.local_group_rank,
        phase=materialized.phase,
        payload_specs=materialized.payload_specs,
        batches=(broken_batch,),
        rank_map=materialized.rank_map,
        expected_outgoing_rows=materialized.expected_outgoing_rows,
        expected_incoming_rows=materialized.expected_incoming_rows,
        logical_plan_digest=materialized.logical_plan_digest,
        published_plan_digest=materialized.published_plan_digest,
        layout_digest=materialized.layout_digest,
        materialized_plan_digest="pending",
        metadata=materialized.metadata,
    )
    broken_plan = MaterializedPlan(
        publication_slot=broken_plan.publication_slot,
        local_global_rank=broken_plan.local_global_rank,
        local_group_rank=broken_plan.local_group_rank,
        phase=broken_plan.phase,
        payload_specs=broken_plan.payload_specs,
        batches=broken_plan.batches,
        rank_map=broken_plan.rank_map,
        expected_outgoing_rows=broken_plan.expected_outgoing_rows,
        expected_incoming_rows=broken_plan.expected_incoming_rows,
        logical_plan_digest=broken_plan.logical_plan_digest,
        published_plan_digest=broken_plan.published_plan_digest,
        layout_digest=broken_plan.layout_digest,
        materialized_plan_digest=broken_plan.recompute_materialized_plan_digest(),
        metadata=broken_plan.metadata,
    )
    validation = CommonPlanValidator().validate(broken_plan, actual_context)
    assert validation.valid is False
    assert validation.reason in {"send_offset_overlap", "recv_offset_overlap", "multiple_outgoing_same_wave", "multiple_incoming_same_wave"}


def test_publisher_rejects_forged_logical_digest() -> None:
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=(0, 1), root_rank=0))
    plan = _build_window_plan()
    published = publisher.build(
        publication_slot={
            "run_id": "run",
            "forward_generation": 0,
            "microbatch_id": "mb",
            "source_layer_id": "0",
            "target_layer_id": "1",
            "planning_slot": "0->1",
        },
        window_plan=plan,
    )
    forged = type(published)(
        publication_slot=published.publication_slot,
        window_plan=published.window_plan,
        logical_plan_digest="forged",
        published_plan_digest=published.published_plan_digest,
        root_global_rank=published.root_global_rank,
        root_group_rank=published.root_group_rank,
        rank_map=published.rank_map,
        version=published.version,
        metadata=published.metadata,
    )
    try:
        publisher.publish(forged)
    except ValueError as exc:
        assert "logical_plan_digest" in str(exc)
    else:
        raise AssertionError("expected forged logical digest to be rejected")


def test_materializer_adds_release_dependencies_for_p1() -> None:
    contexts = make_contexts_from_matrix(phase="P1", matrix=((0, 4), (0, 0)), p2_hint_mode="deterministic_stub")
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=(0, 1), root_rank=0))
    window_plan = WindowPlan(
        planner_id="future:p012:joint:global:rscf",
        planner_family="joint",
        request_digest="0->1:p1",
        waves=(
            PlanWave(
                wave_id=0,
                flows=(
                    PlannedFlow(
                        flow_id="p1_0_1",
                        phase="p1_return",
                        src_rank=0,
                        dst_rank=1,
                        row_count=4,
                        release_state="ready",
                        executable=True,
                    ),
                ),
                estimated_duration=4.0,
            ),
        ),
        metadata={"source_layer_id": "0", "target_layer_id": "1"},
    )
    published = publisher.build(
        publication_slot={
            "run_id": "run",
            "forward_generation": 0,
            "microbatch_id": "mb",
            "source_layer_id": "0",
            "target_layer_id": "1",
            "planning_slot": "0->1",
        },
        window_plan=window_plan,
    )
    sender_context = ActualPhaseContext(
        layer_id="0",
        phase="P1",
        world_size=2,
        rank_space="global",
        layout_digest=str(contexts[0].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[0].to_dict()},
    )
    receiver_context = ActualPhaseContext(
        layer_id="0",
        phase="P1",
        world_size=2,
        rank_space="global",
        layout_digest=str(contexts[1].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[1].to_dict()},
    )
    sender_plan = CommonPlanMaterializer().materialize(published, sender_context)
    receiver_plan = CommonPlanMaterializer().materialize(published, receiver_context)
    sender_dependencies = {
        dep
        for batch in sender_plan.batches
        for slice_ in batch.slices
        for dep in slice_.dependency_ids
    }
    receiver_dependencies = {
        dep
        for batch in receiver_plan.batches
        for slice_ in batch.slices
        for dep in slice_.dependency_ids
    }
    assert "release:0:p0_inbound_complete:0" in sender_dependencies
    assert receiver_dependencies == set()


def test_materializer_keeps_sparse_collective_batches_for_idle_rank() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 4, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        p2_hint_mode="deterministic_stub",
    )
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=(0, 1, 2, 3), root_rank=0))
    published = publisher.build(
        publication_slot={
            "run_id": "run",
            "forward_generation": 0,
            "microbatch_id": "mb",
            "source_layer_id": "0",
            "target_layer_id": "1",
            "planning_slot": "0->1",
        },
        window_plan=WindowPlan(
            planner_id="future:p012:joint:global:rscf",
            planner_family="joint",
            request_digest="sparse:0->1",
            waves=(
                PlanWave(
                    wave_id=0,
                    flows=(
                        PlannedFlow(
                            flow_id="p0_0_1",
                            phase="p0_dispatch",
                            src_rank=0,
                            dst_rank=1,
                            row_count=4,
                            release_state="ready",
                            executable=True,
                        ),
                    ),
                    estimated_duration=4.0,
                ),
            ),
            metadata={"source_layer_id": "0", "target_layer_id": "1"},
        ),
    )
    actual_context = ActualPhaseContext(
        layer_id="0",
        phase="P0",
        world_size=4,
        rank_space="global",
        layout_digest=str(contexts[2].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[2].to_dict()},
    )
    materialized = CommonPlanMaterializer().materialize(published, actual_context)
    assert len(materialized.batches) == 1
    assert materialized.batches[0].collective_required is True
    assert materialized.batches[0].slices == ()
