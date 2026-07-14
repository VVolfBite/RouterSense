from __future__ import annotations

import torch

from rs.core.contracts.execution import ActualPhaseContext
from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan
from rs.runtime.online.megatron_ep.control.plan_publisher import CanonicalPlanPublisher
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.core.contracts.execution import ExecutionContext
from rs.runtime.online.megatron_ep.execution.api import CommonExecutionGuard, PayloadInvocation, PhaseSyncExecutor
from rs.runtime.online.megatron_ep.execution.pipeline import RuntimeExecutionPipeline
from rs.runtime.online.megatron_ep.materialization import CommonPlanMaterializer
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix
from tests.unit.test_plan_materialization import _build_materialized_plan


def _build_local_only_materialized_plan():
    contexts = make_contexts_from_matrix(phase="P0", matrix=((4,),), p2_hint_mode="deterministic_stub")
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=(0,), root_rank=0))
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
            planner_id="barrier_criticality_joint",
            planner_family="joint",
            request_digest="0->1",
            waves=(
                PlanWave(
                    wave_id=0,
                    flows=(
                        PlannedFlow(
                            flow_id="p0_0_0",
                            phase="p0_dispatch",
                            src_rank=0,
                            dst_rank=0,
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
        world_size=1,
        rank_space="global",
        layout_digest=str(contexts[0].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[0].to_dict()},
    )
    return CommonPlanMaterializer().materialize(published, actual_context)


def test_execution_guard_rejects_layer_mismatch() -> None:
    _, _, materialized = _build_materialized_plan()
    guard = CommonExecutionGuard().validate(
        plan=materialized,
        invocation=PayloadInvocation(
            run_id="run",
            forward_generation=0,
            layer_id="99",
            phase="P0",
            payload_role=materialized.payload_specs[0].payload_role,
            shape=(materialized.payload_specs[0].row_count, materialized.payload_specs[0].shape_suffix[0]),
            dtype=materialized.payload_specs[0].dtype,
            layout_digest=materialized.layout_digest,
            invocation_id="inv-0",
            input_tensor=torch.zeros(
                (materialized.payload_specs[0].row_count, materialized.payload_specs[0].shape_suffix[0]),
                dtype=torch.float16,
            ),
        ),
        context=ExecutionContext(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            rank_space="global",
        ),
    )
    assert guard.valid is False
    assert guard.reason == "layer_id_mismatch"


def test_phase_sync_executor_returns_execution_outcome() -> None:
    materialized = _build_local_only_materialized_plan()
    spec = materialized.payload_specs[0]
    rows = spec.row_count
    hidden_dim = spec.shape_suffix[0] if spec.shape_suffix else 1
    tensor = torch.arange(max(rows, 1) * max(hidden_dim, 1), dtype=torch.float16).reshape(max(rows, 1), max(hidden_dim, 1))[:rows]
    outcome = PhaseSyncExecutor().execute(
        plan=materialized,
        invocation=PayloadInvocation(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            payload_role=spec.payload_role,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=spec.dtype,
            layout_digest=materialized.layout_digest,
            invocation_id="inv-1",
            input_tensor=tensor,
        ),
        context=ExecutionContext(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            rank_space="global",
        ),
    )
    assert outcome.success is True
    assert outcome.executed_batch_count == len(materialized.batches)
    assert outcome.completed_task_ids


def test_execution_guard_rejects_duplicate_invocation() -> None:
    _, _, materialized = _build_materialized_plan()
    spec = materialized.payload_specs[0]
    tensor = torch.zeros((spec.row_count, spec.shape_suffix[0]), dtype=torch.float16)
    guard = CommonExecutionGuard()
    invocation = PayloadInvocation(
        run_id="run",
        forward_generation=0,
        layer_id="0",
        phase="P0",
        payload_role=spec.payload_role,
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=spec.dtype,
        layout_digest=materialized.layout_digest,
        invocation_id="dup-invocation",
        input_tensor=tensor,
    )
    context = ExecutionContext(
        run_id="run",
        forward_generation=0,
        layer_id="0",
        phase="P0",
        rank_space="global",
    )
    first = guard.validate(plan=materialized, invocation=invocation, context=context)
    second = guard.validate(plan=materialized, invocation=invocation, context=context)
    assert first.valid is True
    assert second.valid is False
    assert second.reason == "duplicate_invocation"


def test_execution_guard_rejects_generation_mismatch() -> None:
    _, _, materialized = _build_materialized_plan()
    spec = materialized.payload_specs[0]
    tensor = torch.zeros((spec.row_count, spec.shape_suffix[0]), dtype=torch.float16)
    result = CommonExecutionGuard().validate(
        plan=materialized,
        invocation=PayloadInvocation(
            run_id="run",
            forward_generation=999,
            layer_id="0",
            phase="P0",
            payload_role=spec.payload_role,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=spec.dtype,
            layout_digest=materialized.layout_digest,
            invocation_id="bad-generation",
            input_tensor=tensor,
        ),
        context=ExecutionContext(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            rank_space="global",
        ),
    )
    assert result.valid is False
    assert result.reason == "forward_generation_mismatch"


def test_p2p_executor_reports_inflight_batches() -> None:
    materialized = _build_local_only_materialized_plan()
    spec = materialized.payload_specs[0]
    tensor = torch.zeros((spec.row_count, spec.shape_suffix[0]), dtype=torch.float16)
    outcome = PhaseSyncExecutor().execute(
        plan=materialized,
        invocation=PayloadInvocation(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            payload_role=spec.payload_role,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=spec.dtype,
            layout_digest=materialized.layout_digest,
            invocation_id="phase-sync-reference",
            input_tensor=tensor,
        ),
        context=ExecutionContext(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            rank_space="global",
        ),
    )
    from rs.runtime.online.megatron_ep.execution.api import P2PReleaseExecutor

    p2p = P2PReleaseExecutor().execute(
        plan=materialized,
        invocation=PayloadInvocation(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            payload_role=spec.payload_role,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=spec.dtype,
            layout_digest=materialized.layout_digest,
            invocation_id="p2p-reference",
            input_tensor=tensor,
        ),
        context=ExecutionContext(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            rank_space="global",
            metadata={"max_inflight_batches": 1},
        ),
    )
    assert p2p.success is True
    assert p2p.completed_task_ids == outcome.completed_task_ids
    assert p2p.details["max_inflight_batches"] == 1
    assert len(p2p.details["completed_batches"]) == len(materialized.batches)


def test_runtime_execution_pipeline_reservation_rejects_duplicate_invocation() -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((4,),), p2_hint_mode="deterministic_stub")
    publisher = CanonicalPlanPublisher(rank_map=RankMap(group_ranks=(0,), root_rank=0))
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
            planner_id="barrier_criticality_joint",
            planner_family="joint",
            request_digest="0->1",
            waves=(
                PlanWave(
                    wave_id=0,
                    flows=(
                        PlannedFlow(
                            flow_id="p0_0_0",
                            phase="p0_dispatch",
                            src_rank=0,
                            dst_rank=0,
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
        world_size=1,
        rank_space="global",
        layout_digest=str(contexts[0].canonical_receive_layout_id),
        metadata={"phase_ready_context": contexts[0].to_dict()},
    )
    pipeline = RuntimeExecutionPipeline()
    prepared = pipeline.prepare(published, actual_context)
    spec = prepared.materialized_plan.payload_specs[0]
    tensor = torch.arange(spec.row_count * spec.shape_suffix[0], dtype=torch.float16).reshape(spec.row_count, spec.shape_suffix[0])
    invocation = PayloadInvocation(
        run_id="run",
        forward_generation=0,
        layer_id="0",
        phase="P0",
        payload_role=spec.payload_role,
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=spec.dtype,
        layout_digest=prepared.materialized_plan.layout_digest,
        invocation_id="dup-through-pipeline",
        input_tensor=tensor,
    )
    context = ExecutionContext(
        run_id="run",
        forward_generation=0,
        layer_id="0",
        phase="P0",
        rank_space="global",
    )
    first = pipeline.execute(prepared, invocation, context)
    second = pipeline.execute(prepared, invocation, context)
    assert first.success is True
    assert second.success is False
    assert second.failure_code == "duplicate_invocation"
