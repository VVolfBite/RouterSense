from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts.execution import ActualPhaseContext, ExecutionContext, ExecutionOutcome, MaterializedPlan, PublishedPlan, ValidationResult
from rs.runtime.online.megatron_ep.execution.api import (
    CommonExecutionGuard,
    GlooFunctionalExecutor,
    NativePassthroughExecutor,
    P2PReleaseExecutor,
    PayloadInvocation,
    PhaseSyncExecutor,
    _validate_output_tensor,
)
from rs.runtime.online.megatron_ep.materialization import CommonPlanMaterializer, CommonPlanValidator


@dataclass(frozen=True)
class PreparedExecution:
    published_plan: PublishedPlan
    actual_phase_context: ActualPhaseContext
    materialized_plan: MaterializedPlan
    validation: ValidationResult


class RuntimeExecutionPipeline:
    def __init__(
        self,
        *,
        materializer: CommonPlanMaterializer | None = None,
        validator: CommonPlanValidator | None = None,
        guard: CommonExecutionGuard | None = None,
        executor: PhaseSyncExecutor | None = None,
    ) -> None:
        self._materializer = materializer or CommonPlanMaterializer()
        self._validator = validator or CommonPlanValidator()
        self._guard = guard or CommonExecutionGuard()
        self._executor = executor or PhaseSyncExecutor()

    def prepare(self, published_plan: PublishedPlan, actual_phase_context: ActualPhaseContext) -> PreparedExecution:
        materialized = self._materializer.materialize(published_plan, actual_phase_context)
        validation = self._validator.validate(materialized, actual_phase_context)
        return PreparedExecution(
            published_plan=published_plan,
            actual_phase_context=actual_phase_context,
            materialized_plan=materialized,
            validation=validation,
        )

    def execute(self, prepared_execution: PreparedExecution, invocation: PayloadInvocation, context: ExecutionContext) -> ExecutionOutcome:
        if not bool(prepared_execution.validation.valid):
            return ExecutionOutcome(
                success=False,
                output_payload=None,
                submitted_task_ids=tuple(),
                completed_task_ids=tuple(),
                failed_task_ids=tuple(),
                unresolved_task_ids=tuple(),
                executed_batch_count=0,
                all_work_completed=False,
                failure_code="prepared_execution_invalid",
                details={
                    "stage": str(prepared_execution.validation.stage),
                    "reason": str(prepared_execution.validation.reason),
                    "details": dict(prepared_execution.validation.details),
                },
            )
        guard = self._guard.reserve(plan=prepared_execution.materialized_plan, invocation=invocation, context=context)
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
                details={"stage": str(guard.stage)},
            )
        try:
            outcome = self._executor.execute(plan=prepared_execution.materialized_plan, invocation=invocation, context=context)
        except Exception:
            self._guard.rollback(invocation=invocation, context=context)
            raise
        failure = _validate_execution_outcome(
            outcome=outcome,
            plan=prepared_execution.materialized_plan,
            invocation=invocation,
        )
        if failure is not None:
            self._guard.rollback(invocation=invocation, context=context)
            return failure
        self._guard.commit(invocation=invocation, context=context)
        return outcome


def build_runtime_execution_pipeline(*, execution_mode: str) -> RuntimeExecutionPipeline:
    normalized = str(execution_mode or "").strip().lower()
    if normalized in {"phase_sync_wave", "multiphase_pending_window", "phase_sync"}:
        executor = PhaseSyncExecutor()
    elif normalized in {"joint_window_async_p2p", "async_release"}:
        executor = P2PReleaseExecutor()
    elif normalized == "gloo_functional":
        executor = GlooFunctionalExecutor()
    elif normalized in {"native_passthrough", "native", "disabled"}:
        executor = NativePassthroughExecutor()
    else:
        raise ValueError(f"unsupported execution_mode {execution_mode!r}")
    return RuntimeExecutionPipeline(executor=executor)


def _validate_execution_outcome(
    *,
    outcome: ExecutionOutcome,
    plan: MaterializedPlan,
    invocation: PayloadInvocation,
) -> ExecutionOutcome | None:
    try:
        outcome.validate()
    except Exception as exc:
        return ExecutionOutcome(
            success=False,
            output_payload=None,
            submitted_task_ids=tuple(),
            completed_task_ids=tuple(),
            failed_task_ids=tuple(),
            unresolved_task_ids=tuple(),
            executed_batch_count=0,
            all_work_completed=False,
            failure_code="invalid_execution_outcome",
            details={"reason": f"{type(exc).__name__}: {exc}", "backend_failure_code": outcome.failure_code},
        )
    submitted = tuple(str(value) for value in outcome.submitted_task_ids)
    completed = tuple(str(value) for value in outcome.completed_task_ids)
    failed = tuple(str(value) for value in outcome.failed_task_ids)
    unresolved = tuple(str(value) for value in outcome.unresolved_task_ids)
    submitted_set = set(submitted)
    completed_set = set(completed)
    failed_set = set(failed)
    unresolved_set = set(unresolved)
    expected_submitted = {
        str(item.task_id)
        for batch in plan.batches
        for item in batch.slices
        if str(item.payload_role) == str(invocation.payload_role)
    }
    if len(submitted_set) != len(submitted):
        reason = "duplicate_submitted_task_ids"
    elif len(completed_set) != len(completed):
        reason = "duplicate_completed_task_ids"
    elif len(failed_set) != len(failed):
        reason = "duplicate_failed_task_ids"
    elif len(unresolved_set) != len(unresolved):
        reason = "duplicate_unresolved_task_ids"
    elif not completed_set.issubset(submitted_set):
        reason = "completed_not_subset_of_submitted"
    elif not failed_set.issubset(submitted_set):
        reason = "failed_not_subset_of_submitted"
    elif not unresolved_set.issubset(submitted_set):
        reason = "unresolved_not_subset_of_submitted"
    elif completed_set & failed_set or completed_set & unresolved_set or failed_set & unresolved_set:
        reason = "task_sets_not_disjoint"
    elif submitted_set != expected_submitted:
        reason = "submitted_task_ids_mismatch"
    elif outcome.success and (
        not bool(outcome.all_work_completed)
        or completed_set != expected_submitted
        or failed_set
        or unresolved_set
        or outcome.output_payload is None
    ):
        reason = "successful_outcome_incomplete"
    elif not outcome.success and not str(outcome.failure_code or ""):
        reason = "failure_code_missing"
    elif outcome.success:
        output_reason = _validate_output_tensor(plan, invocation, outcome.output_payload if hasattr(outcome, "output_payload") else None)
        if output_reason is not None:
            reason = output_reason
        else:
            reason = ""
    else:
        reason = ""
    if not reason:
        return None
    return ExecutionOutcome(
        success=False,
        output_payload=None,
        submitted_task_ids=submitted,
        completed_task_ids=completed,
        failed_task_ids=failed,
        unresolved_task_ids=unresolved if unresolved else tuple(task_id for task_id in submitted if task_id not in completed_set),
        executed_batch_count=int(outcome.executed_batch_count),
        all_work_completed=False,
        failure_code="invalid_execution_outcome",
        details={"reason": reason, "backend_failure_code": outcome.failure_code, "backend_details": dict(outcome.details)},
    )
