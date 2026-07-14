from __future__ import annotations

import torch

from rs.core.contracts.execution import ExecutionContext
from rs.runtime.online.megatron_ep.execution.api import CommonExecutionGuard, PayloadInvocation, PhaseSyncExecutor
from tests.unit.test_plan_materialization import _build_materialized_plan


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
    _, _, materialized = _build_materialized_plan()
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
    _, _, materialized = _build_materialized_plan()
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
