M2 formal execution closure

This branch now provides a formal M2 execution chain that is independent from the old
`rs.scheduling.phase_execution` path for publication, materialization, validation, guard,
and executor contracts.

Implemented:

- `rs.core.contracts.execution` now defines typed `PublishedPlan`, `MaterializedPlan`,
  `ExecutionContext`, `ExecutionOutcome`, and aligned `PlanPublisher` / `PlanMaterializer`
  / `PlanValidator` / `ExecutionGuard` / `Executor` protocols.
- `CanonicalPlanPublisher` binds a real `WindowPlan`, recomputes logical and published
  digests, and rejects forged logical digests.
- `CommonPlanMaterializer` materializes directly from `PublishedPlan(WindowPlan)` and
  `ActualPhaseContext` without routing through legacy abstract phase-execution plans.
- `CommonPlanValidator` validates:
  - published/materialized digest binding
  - outgoing and incoming row coverage
  - send/recv offset gaps
  - send/recv offset overlap
  - payload-role consistency
- `CommonExecutionGuard` validates invocation identity, generation, layer, phase,
  payload role, dtype, layout digest, and duplicate invocation IDs.
- `PhaseSyncExecutor`, `P2PReleaseExecutor`, and `GlooFunctionalExecutor` share one
  typed executor protocol and emit `ExecutionOutcome` from execution-path evidence.
- `RuntimeExecutionPipeline` is the single formal M2 integration entry for
  prepare/materialize/validate and guard/execute.
- formal 4-rank Gloo gate added in
  `RS/experiments/distributed/run_m2_formal_execution_gloo.py`

Executed commands:

- `python -m compileall RS/src/rs/core/contracts/execution.py RS/src/rs/runtime/online/megatron_ep/control/plan_publisher.py RS/src/rs/runtime/online/megatron_ep/materialization RS/src/rs/runtime/online/megatron_ep/execution RS/tests/unit/test_rank_map.py RS/tests/unit/test_plan_materialization.py RS/tests/unit/test_execution_api.py RS/tests/test_architecture_dependencies.py RS/experiments/distributed/run_m2_formal_execution_gloo.py`
- `PYTHONPATH='RS/src;RS;.' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q RS/tests/unit/test_rank_map.py RS/tests/unit/test_plan_materialization.py RS/tests/unit/test_execution_api.py RS/tests/test_architecture_dependencies.py`
- `PYTHONPATH='RS/src;RS;.' python RS/experiments/distributed/run_m2_formal_execution_gloo.py`

Observed results:

- compileall passed
- focused M2 tests passed: `31 passed`
- formal 4-rank Gloo gate passed for:
  - full group `(0,1,2,3)`
  - non-contiguous subgroup execution probe `(2,3)`
  - direct `WindowPlan -> PublishedPlan -> MaterializedPlan -> Executor` path

Residual notes:

- legacy runtime surfaces such as `executor_facade`, `transport_adapter`,
  `plan_agreement`, and `compiler_facade` still exist in the repo for compatibility.
  They are no longer the formal M2 owner, but final runtime ownership is established in
  `convergence/m123-integration`.
- this branch does not modify `lifecycle.py`; formal runtime call-site wiring remains an
  integration responsibility by design.

Branch status:

- `M2_CONTROL_MATERIALIZATION_EXECUTION_READY`
