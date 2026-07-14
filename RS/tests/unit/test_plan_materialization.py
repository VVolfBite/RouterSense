from __future__ import annotations

import pytest

from rs.core.contracts.execution import ActualPhaseContext
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.runtime.online.megatron_ep.materialization import CommonPlanMaterializer, CommonPlanValidator
from rs.scheduling.registry import resolve_phase_policy
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def _build_materialized_plan():
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 48), (32, 0)), p2_hint_mode="deterministic_stub")
    policy = resolve_phase_policy(policy_name="greedy_ready_set", bucket_rows=16)
    full_plan = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=(0, 1), root_rank=0))
    published = publisher.build(
        planner_id="greedy_ready_set",
        logical_plan=full_plan.to_abstract_plan().to_dict(),
        logical_plan_digest=str(full_plan.plan_hash),
        publication_slot_digest="slot-0",
        metadata={},
    )
    actual_context = ActualPhaseContext(
        layer_id="0",
        phase="P0",
        world_size=2,
        rank_space="global",
        layout_digest=str(contexts[0].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[0].to_dict(), "local_rank": 0},
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
    broken_batch = type(batch)(
        batch_id=batch.batch_id,
        wave_id=batch.wave_id,
        slices=(type(broken_slice)(
            flow_id=broken_slice.flow_id,
            task_id=broken_slice.task_id,
            payload_role="unexpected_role",
            src_rank=broken_slice.src_rank,
            dst_rank=broken_slice.dst_rank,
            row_count=broken_slice.row_count,
            send_offset_rows=broken_slice.send_offset_rows,
            recv_offset_rows=broken_slice.recv_offset_rows,
            dependency_ids=broken_slice.dependency_ids,
        ),) + batch.slices[1:],
        metadata=batch.metadata,
    )
    broken_plan = type(materialized)(
        published_plan_digest=materialized.published_plan_digest,
        materialized_plan_digest=materialized.materialized_plan_digest,
        layout_digest=materialized.layout_digest,
        payload_specs=materialized.payload_specs,
        batches=(broken_batch,) + materialized.batches[1:],
        local_rank=materialized.local_rank,
        layer_id=materialized.layer_id,
        phase=materialized.phase,
        logical_plan_digest=materialized.logical_plan_digest,
        expected_payload_roles=materialized.expected_payload_roles,
        metadata=materialized.metadata,
    )
    validation = CommonPlanValidator().validate(broken_plan, actual_context)
    assert validation.valid is False

