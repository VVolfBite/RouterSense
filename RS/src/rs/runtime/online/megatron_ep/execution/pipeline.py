from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts.execution import ActualPhaseContext, ExecutionContext, ExecutionOutcome, MaterializedPlan, PublishedPlan, ValidationResult
from rs.runtime.online.megatron_ep.execution.api import CommonExecutionGuard, PayloadInvocation, PhaseSyncExecutor
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
            self._guard.rollback(str(invocation.invocation_id))
            raise
        self._guard.commit(str(invocation.invocation_id))
        return outcome
