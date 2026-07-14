from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from rs.core.contracts.execution import ExecutionContext, ExecutionOutcome, MaterializedPlan, PayloadSpec, ValidationResult
from rs.runtime.online.megatron_ep.execution.executor_facade import ExecutionRequest, execute_transport
from rs.runtime.online.megatron_ep.phase import PhaseExecutionPlan, PhaseReadyContext
from rs.scheduling.validation import stable_hash


def _coerce_phase_ready_context(payload: object) -> PhaseReadyContext:
    if isinstance(payload, PhaseReadyContext):
        return payload
    if isinstance(payload, dict):
        return PhaseReadyContext.from_dict(dict(payload))
    raise ValueError("phase_ready_context metadata is required")


def _coerce_execution_plan(payload: object) -> PhaseExecutionPlan:
    if isinstance(payload, PhaseExecutionPlan):
        return payload
    if isinstance(payload, dict):
        return PhaseExecutionPlan.from_dict(dict(payload))
    raise ValueError("phase_execution_plan metadata is required")


class CommonExecutionGuard:
    def validate(self, plan: MaterializedPlan, context: ExecutionContext) -> ValidationResult:
        try:
            plan.validate()
            context.validate()
        except Exception as exc:
            return ValidationResult(valid=False, stage="guard", reason=str(exc))
        if str(plan.layer_id) != str(context.layer_id):
            return ValidationResult(valid=False, stage="guard", reason="layer_id_mismatch")
        if str(plan.phase) != str(context.phase):
            return ValidationResult(valid=False, stage="guard", reason="phase_mismatch")
        metadata = dict(plan.metadata)
        phase_ready_context = metadata.get("phase_ready_context")
        if phase_ready_context is None:
            return ValidationResult(valid=False, stage="guard", reason="missing_phase_ready_context")
        return ValidationResult(valid=True, stage="guard")


@dataclass(frozen=True)
class PayloadInvocation:
    payload_role: str
    input_tensor: torch.Tensor
    process_group: dist.ProcessGroup | None = None
    rank_context: dict[str, Any] = field(default_factory=dict)
    event_sink: Any | None = None


class _BaseExecutor:
    backend_id = ""

    def execute(self, plan: MaterializedPlan, payload: PayloadInvocation, context: ExecutionContext) -> ExecutionOutcome:
        guard = CommonExecutionGuard().validate(plan, context)
        if not guard.valid:
            return ExecutionOutcome(
                success=False,
                executed_batch_count=0,
                all_work_completed=False,
                execution_digest="",
                unresolved_task_ids=tuple(),
                details={"reason": str(guard.reason), "stage": str(guard.stage)},
            )
        metadata = dict(plan.metadata)
        phase_ready_context = _coerce_phase_ready_context(metadata["phase_ready_context"])
        phase_execution_plan = _coerce_execution_plan(metadata["phase_execution_plan"])
        result = execute_transport(
            ExecutionRequest(
                execution_plan=phase_execution_plan,
                phase_context=phase_ready_context,
                tensor_role=str(payload.payload_role),
                input_tensor=payload.input_tensor,
                process_group=payload.process_group,
                rank_context=dict(payload.rank_context),
                event_sink=payload.event_sink,
                requested_backend_id=str(self.backend_id),
            ),
            backend=str(self.backend_id),
        )
        completed_task_ids = tuple(
            str(item.task_id)
            for batch in plan.batches
            for item in batch.slices
            if str(item.payload_role) == str(payload.payload_role)
        )
        unresolved = () if bool(result.all_work_completed) else completed_task_ids
        digest = stable_hash(
            {
                "backend_id": str(self.backend_id),
                "published_plan_digest": str(plan.published_plan_digest),
                "materialized_plan_digest": str(plan.materialized_plan_digest),
                "payload_role": str(payload.payload_role),
                "completed_task_ids": list(completed_task_ids),
                "all_work_completed": bool(result.all_work_completed),
            }
        )
        return ExecutionOutcome(
            success=bool(result.preflight_passed and result.all_work_completed and not result.timeout),
            executed_batch_count=int(len(plan.batches)),
            all_work_completed=bool(result.all_work_completed),
            execution_digest=str(digest),
            completed_task_ids=completed_task_ids,
            failed_task_ids=tuple(),
            unresolved_task_ids=tuple(unresolved),
            details={
                "backend_id": str(self.backend_id),
                "output_shape": tuple(int(dim) for dim in result.output_tensor.shape),
                "output_dtype": str(result.output_tensor.dtype),
                "send_op_count": int(result.send_op_count),
                "recv_op_count": int(result.recv_op_count),
                "all_work_completed": bool(result.all_work_completed),
                "raw_summary": dict(result.raw_summary),
            },
        )


class PhaseSyncExecutor(_BaseExecutor):
    backend_id = "phase_sync"


class P2PReleaseExecutor(_BaseExecutor):
    backend_id = "async_release"


class GlooFunctionalExecutor(_BaseExecutor):
    backend_id = "phase_sync"
