from __future__ import annotations

from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan
from rs.core.contracts.execution import ActualPhaseContext, MaterializedPlan, PayloadSpec, TransferSlice, ExecutionBatch
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.materialization import CommonPlanMaterializer, CommonPlanValidator
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def _build_window_plan() -> WindowPlan:
    return WindowPlan(
        planner_id="barrier_criticality_joint",
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
        slices=(
            TransferSlice(
                flow_id=broken_slice.flow_id,
                task_id=broken_slice.task_id,
                payload_role="unexpected_role",
                src_rank=broken_slice.src_rank,
                dst_rank=broken_slice.dst_rank,
                row_count=broken_slice.row_count,
                send_offset_rows=broken_slice.send_offset_rows,
                recv_offset_rows=broken_slice.recv_offset_rows,
                dependency_ids=broken_slice.dependency_ids,
            ),
        )
        + batch.slices[1:],
        metadata=batch.metadata,
    )
    broken_plan = MaterializedPlan(
        publication_slot=materialized.publication_slot,
        local_global_rank=materialized.local_global_rank,
        local_group_rank=materialized.local_group_rank,
        phase=materialized.phase,
        payload_specs=materialized.payload_specs,
        batches=(broken_batch,) + materialized.batches[1:],
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
        slices=first_batch.slices[1:],
        metadata=first_batch.metadata,
    )
    broken_plan = MaterializedPlan(
        publication_slot=materialized.publication_slot,
        local_global_rank=materialized.local_global_rank,
        local_group_rank=materialized.local_group_rank,
        phase=materialized.phase,
        payload_specs=materialized.payload_specs,
        batches=(broken_batch,),
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
