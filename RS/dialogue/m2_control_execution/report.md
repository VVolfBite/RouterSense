M2 formal execution closure

This branch now closes the gap between typed M2 contracts and the actual distributed
execution path. The previous branch state had formal contracts, but the executors still
behaved like local clones. The current branch runs the formal chain:

`PublishedPlan(WindowPlan) -> MaterializedPlan -> Validator -> Guard reservation -> Executor`

Implemented

- `PublishedPlan` now owns:
  - typed `PublicationSlot`
  - `RankMapSnapshot`
  - `rank_space="group_local"`
  - recomputed `logical_plan_digest` and `published_plan_digest`
- `TransferSlice` now stores:
  - `src_group_rank` / `dst_group_rank`
  - `src_global_rank` / `dst_global_rank`
- `CommonPlanMaterializer` now:
  - treats `WindowPlan` ranks as EP group-local
  - converts to global ranks through `RankMap`
  - validates peer-local send/recv coverage
  - emits release tokens:
    - `release:p0_inbound_complete:<group-rank>`
    - `release:p1_inbound_complete:<group-rank>`
- `CommonPlanValidator` now validates:
  - peer-local outgoing coverage
  - peer-local incoming coverage
  - send/recv offset gap and overlap
  - digest binding
  - rank-space consistency
- `RuntimeExecutionPipeline` now uses guard reservation with commit/rollback.
- `PhaseSyncExecutor` now performs real `dist.all_to_all_single` transport.
- `PhaseSyncExecutor` now preserves globally ordered sparse collective rounds so zero-flow ranks
  still participate in every formal wave with zero splits.
- `P2PReleaseExecutor` now performs real asynchronous `dist.isend` / `dist.irecv`
  transport, waits actual work handles, and records `peak_inflight_batches`.
- `GlooFunctionalExecutor` follows the same transport core instead of returning a local clone.
- `run_m2_formal_execution_gloo.py` now asserts:
  - real distributed op count > 0
  - sparse-wave collective rounds execute on all ranks, including zero-flow ranks
  - full-group and subgroup rank-space correctness
  - exact expected remote rows
  - PhaseSync output == P2P output
  - `peak_inflight_batches >= 2` when the formal plan has two ready batches

Focused verification

- `python -m compileall RS/src/rs/core/contracts/execution.py RS/src/rs/runtime/online/megatron_ep/control/plan_publisher.py RS/src/rs/runtime/online/megatron_ep/materialization RS/src/rs/runtime/online/megatron_ep/execution RS/tests/unit/test_rank_map.py RS/tests/unit/test_plan_materialization.py RS/tests/unit/test_execution_api.py RS/tests/test_architecture_dependencies.py RS/experiments/distributed/run_m2_formal_execution_gloo.py`
- `PYTHONPATH='RS/src;RS;.' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q RS/tests/unit/test_rank_map.py RS/tests/unit/test_plan_materialization.py RS/tests/unit/test_execution_api.py RS/tests/test_architecture_dependencies.py`
- `PYTHONPATH='RS/src;RS;.' python RS/experiments/distributed/run_m2_formal_execution_gloo.py`

Observed results

- compileall passed
- focused pytest passed: `36 passed`
- formal 4-rank Gloo gate passed for:
  - full group `(0,1,2,3)`
  - sparse wave `0 -> 1` with ranks `2/3` participating via zero splits
  - sparse two-wave execution with collective round count `2` on every rank
  - subgroup `(2,3)` with group-local `WindowPlan` ranks `(0,1)`
  - real remote tensor validation
  - real asynchronous P2P execution with `peak_inflight_batches = 2`

Residual notes

- This branch still does not wire the execution pipeline into `lifecycle.py`. That
  ownership remains with `convergence/m123-integration`.
- Legacy compatibility modules still exist in the tree, but they are no longer the
  formal M2 execution owner.

Branch status

- `M2_CONTROL_MATERIALIZATION_EXECUTION_READY`
