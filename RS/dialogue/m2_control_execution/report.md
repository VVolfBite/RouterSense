M2 partial implementation checkpoint

This branch no longer matches the interface-lock tree.

Implemented in this checkpoint:

- `rs.core.contracts.execution` rewritten around typed `PublishedPlan`, `MaterializedPlan`, `ExecutionOutcome`, and `Executor`
- `CanonicalPlanPublisher` now binds a real `WindowPlan` and recomputes published digests
- `CommonPlanMaterializer` now materializes directly from `PublishedPlan(WindowPlan)` and `ActualPhaseContext`
- `CommonPlanValidator` now validates digest binding plus outgoing/incoming row coverage
- `CommonExecutionGuard` now validates invocation identity and rejects duplicate invocation IDs
- `RuntimeExecutionPipeline` added as the single integration entry for prepare/execute
- focused unit tests updated to exercise direct `WindowPlan` materialization and fail-closed validator behavior

Executed commands:

- `python -m compileall RS/src/rs/core/contracts RS/src/rs/runtime/online/megatron_ep/control RS/src/rs/runtime/online/megatron_ep/materialization RS/src/rs/runtime/online/megatron_ep/execution RS/tests/unit/test_rank_map.py RS/tests/unit/test_plan_materialization.py RS/tests/unit/test_execution_api.py`
- `PYTHONPATH='RS/src;RS;.' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q RS/tests/unit/test_rank_map.py RS/tests/unit/test_plan_materialization.py RS/tests/unit/test_execution_api.py`

Observed results:

- compileall passed
- focused M2 unit tests passed: `10 passed`

Still blocked before `M2_CONTROL_MATERIALIZATION_EXECUTION_READY`:

1. formal runtime call path is not yet wired in `convergence/m123-integration`
2. Gloo 4-rank execution gate is not implemented
3. legacy `phase_execution` / `executor_facade` / `transport_adapter` surfaces still exist and formal path ownership is not yet cut over
4. negative coverage for subgroup, backend failure, and inflight P2P semantics is still incomplete
