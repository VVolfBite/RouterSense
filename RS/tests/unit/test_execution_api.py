from __future__ import annotations

import torch

from rs.core.contracts.execution import ExecutionContext
from rs.runtime.online.megatron_ep.execution.api import CommonExecutionGuard, PayloadInvocation, PhaseSyncExecutor
from tests.unit.test_plan_materialization import _build_materialized_plan


def test_execution_guard_rejects_layer_mismatch() -> None:
    _, _, materialized = _build_materialized_plan()
    guard = CommonExecutionGuard().validate(
        materialized,
        ExecutionContext(
            run_id="run",
            forward_generation=0,
            layer_id="99",
            phase="P0",
            rank_space="global",
        ),
    )
    assert guard.valid is False
    assert guard.reason == "layer_id_mismatch"


def test_phase_sync_executor_returns_execution_outcome(monkeypatch) -> None:
    _, _, materialized = _build_materialized_plan()
    payload_role = materialized.payload_specs[0].payload_role
    rows = materialized.payload_specs[0].row_count
    hidden_dim = materialized.payload_specs[0].shape_suffix[0] if materialized.payload_specs[0].shape_suffix else 1
    tensor = torch.arange(max(rows, 1) * max(hidden_dim, 1), dtype=torch.float16).reshape(max(rows, 1), max(hidden_dim, 1))[:rows]
    from rs.runtime.online.megatron_ep.execution.executor_facade import ExecutionResult

    monkeypatch.setattr(
        "rs.runtime.online.megatron_ep.execution.api.execute_transport",
        lambda request, backend: ExecutionResult(
            output_tensor=request.input_tensor.clone(),
            execution_plan_digest="plan-digest",
            send_op_count=1,
            recv_op_count=1,
            local_copy_task_count=0,
            local_copy_row_count=0,
            enqueue_us=1.0,
            wait_us=2.0,
            total_us=3.0,
            fallback_used=False,
            timeout=False,
            raw_summary={},
            execution_entries=(),
            requested_backend_id=backend,
            backend_id=backend,
            executed_backend_id=backend,
            all_work_completed=True,
        ),
    )
    outcome = PhaseSyncExecutor().execute(
        materialized,
        PayloadInvocation(payload_role=payload_role, input_tensor=tensor),
        ExecutionContext(
            run_id="run",
            forward_generation=0,
            layer_id="0",
            phase="P0",
            rank_space="global",
        ),
    )
    assert outcome.execution_digest
    assert outcome.executed_batch_count == len(materialized.batches)
