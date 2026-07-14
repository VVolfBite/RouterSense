from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from rs.core.contracts.execution import ExecutionContext, ExecutionOutcome, MaterializedPlan, ValidationResult
from rs.runtime.online.megatron_ep.phase import PhaseReadyContext


def _coerce_phase_ready_context(payload: object) -> PhaseReadyContext:
    if isinstance(payload, PhaseReadyContext):
        return payload
    if isinstance(payload, dict):
        return PhaseReadyContext.from_dict(dict(payload))
    raise ValueError("phase_ready_context metadata is required")


@dataclass(frozen=True)
class PayloadInvocation:
    run_id: str
    forward_generation: int
    layer_id: str
    phase: str
    payload_role: str
    shape: tuple[int, ...]
    dtype: str
    layout_digest: str
    invocation_id: str
    input_tensor: torch.Tensor | None = None
    process_group: dist.ProcessGroup | None = None
    rank_context: dict[str, Any] = field(default_factory=dict)
    event_sink: Any | None = None

    def validate(self) -> None:
        if not str(self.run_id):
            raise ValueError("run_id must be non-empty")
        if int(self.forward_generation) < 0:
            raise ValueError("forward_generation must be >= 0")
        if not str(self.layer_id) or not str(self.phase) or not str(self.payload_role):
            raise ValueError("payload invocation identity must be non-empty")
        if not str(self.dtype):
            raise ValueError("dtype must be non-empty")
        if not str(self.layout_digest):
            raise ValueError("layout_digest must be non-empty")
        if not str(self.invocation_id):
            raise ValueError("invocation_id must be non-empty")
        for dim in self.shape:
            if int(dim) < 0:
                raise ValueError("shape dims must be >= 0")


class CommonExecutionGuard:
    def __init__(self) -> None:
        self._consumed_invocations: set[str] = set()

    def validate(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ValidationResult:
        try:
            plan.validate()
            invocation.validate()
            context.validate()
        except Exception as exc:
            return ValidationResult(valid=False, stage="guard", reason=str(exc))
        if str(context.run_id) != str(invocation.run_id):
            return ValidationResult(valid=False, stage="guard", reason="run_id_mismatch")
        if int(context.forward_generation) != int(invocation.forward_generation):
            return ValidationResult(valid=False, stage="guard", reason="forward_generation_mismatch")
        if str(context.layer_id) != str(invocation.layer_id):
            return ValidationResult(valid=False, stage="guard", reason="layer_id_mismatch")
        if str(context.phase) != str(invocation.phase):
            return ValidationResult(valid=False, stage="guard", reason="phase_mismatch")
        if str(plan.phase) != str(invocation.phase):
            return ValidationResult(valid=False, stage="guard", reason="plan_phase_mismatch")
        if str(plan.layout_digest) != str(invocation.layout_digest):
            return ValidationResult(valid=False, stage="guard", reason="layout_digest_mismatch")
        roles = {str(item.payload_role) for item in plan.payload_specs}
        if str(invocation.payload_role) not in roles:
            return ValidationResult(valid=False, stage="guard", reason="payload_role_mismatch")
        matching_spec = next(item for item in plan.payload_specs if str(item.payload_role) == str(invocation.payload_role))
        if str(invocation.dtype) != str(matching_spec.dtype):
            return ValidationResult(valid=False, stage="guard", reason="dtype_mismatch")
        if invocation.input_tensor is not None and tuple(int(dim) for dim in invocation.input_tensor.shape) != tuple(int(dim) for dim in invocation.shape):
            return ValidationResult(valid=False, stage="guard", reason="shape_mismatch")
        if str(invocation.invocation_id) in self._consumed_invocations:
            return ValidationResult(valid=False, stage="guard", reason="duplicate_invocation")
        self._consumed_invocations.add(str(invocation.invocation_id))
        return ValidationResult(valid=True, stage="guard")


class _BaseExecutor:
    backend_id = ""

    def _submitted_task_ids(self, plan: MaterializedPlan, payload_role: str) -> tuple[str, ...]:
        return tuple(
            str(item.task_id)
            for batch in plan.batches
            for item in batch.slices
            if str(item.payload_role) == str(payload_role)
        )

    def execute(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ExecutionOutcome:
        guard = CommonExecutionGuard().validate(plan=plan, invocation=invocation, context=context)
        if not guard.valid:
            return ExecutionOutcome(
                success=False,
                output_payload=None,
                submitted_task_ids=tuple(),
                completed_task_ids=tuple(),
                failed_task_ids=tuple(),
                unresolved_task_ids=tuple(),
                executed_batch_count=0,
                all_work_completed=False,
                failure_code=str(guard.reason or "guard_failed"),
                details={"stage": str(guard.stage), "reason": str(guard.reason)},
            )
        if invocation.input_tensor is None:
            return ExecutionOutcome(
                success=False,
                output_payload=None,
                submitted_task_ids=tuple(),
                completed_task_ids=tuple(),
                failed_task_ids=tuple(),
                unresolved_task_ids=tuple(),
                executed_batch_count=0,
                all_work_completed=False,
                failure_code="missing_input_tensor",
                details={},
            )
        submitted_task_ids = self._submitted_task_ids(plan, invocation.payload_role)
        if not submitted_task_ids:
            return ExecutionOutcome(
                success=True,
                output_payload=invocation.input_tensor.clone(),
                submitted_task_ids=tuple(),
                completed_task_ids=tuple(),
                failed_task_ids=tuple(),
                unresolved_task_ids=tuple(),
                executed_batch_count=0,
                all_work_completed=True,
                details={"backend_id": str(self.backend_id), "reason": "no_matching_payload_role"},
            )
        _coerce_phase_ready_context(dict(plan.metadata).get("phase_ready_context"))
        completed_task_ids = submitted_task_ids
        return ExecutionOutcome(
            success=True,
            output_payload=invocation.input_tensor.clone(),
            submitted_task_ids=submitted_task_ids,
            completed_task_ids=completed_task_ids,
            failed_task_ids=tuple(),
            unresolved_task_ids=tuple(),
            executed_batch_count=int(len(plan.batches)),
            all_work_completed=True,
            failure_code=None,
            details={
                "backend_id": str(self.backend_id),
                "submitted_task_count": int(len(submitted_task_ids)),
            },
        )


class PhaseSyncExecutor(_BaseExecutor):
    backend_id = "phase_sync"


class P2PReleaseExecutor(_BaseExecutor):
    backend_id = "async_release"

    def execute(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ExecutionOutcome:
        submitted = self._submitted_task_ids(plan, invocation.payload_role)
        if not submitted:
            return super().execute(plan=plan, invocation=invocation, context=context)
        max_inflight_batches = int(dict(context.metadata).get("max_inflight_batches", 2) or 2)
        inflight: deque[str] = deque()
        completed_batches: list[str] = []
        for batch in plan.batches:
            inflight.append(str(batch.batch_id))
            if len(inflight) >= max_inflight_batches:
                completed_batches.append(inflight.popleft())
        while inflight:
            completed_batches.append(inflight.popleft())
        base = super().execute(plan=plan, invocation=invocation, context=context)
        return ExecutionOutcome(
            success=base.success,
            output_payload=base.output_payload,
            submitted_task_ids=base.submitted_task_ids,
            completed_task_ids=base.completed_task_ids,
            failed_task_ids=base.failed_task_ids,
            unresolved_task_ids=base.unresolved_task_ids,
            executed_batch_count=base.executed_batch_count,
            all_work_completed=base.all_work_completed,
            failure_code=base.failure_code,
            details={**dict(base.details), "completed_batches": tuple(completed_batches), "max_inflight_batches": max_inflight_batches},
        )


class GlooFunctionalExecutor(_BaseExecutor):
    backend_id = "gloo_functional"
