**Status**

Phase A is implemented and CPU-validated.

Current branch status:

- `M4_OFFLINE_CORE_READY`
- overall `M4_OFFLINE_EVALUATION_BLOCKED`

**What changed**

1. Added canonical offline contracts in `src/rs/core/contracts/offline.py`.
2. Added formal offline package `src/rs/offline/`.
3. Added formal simulation namespace `src/rs/simulation/`.
4. Switched `runtime/offline/replay_unified.py` to build formal `PlanningRequest` through `OfflinePlanningRequestBuilder` and to evaluate realized makespan through `OfflineEvaluator`.
5. Preserved legacy replay audit only as compatibility evidence.

**Single builder**

Formal builder is now:

- `rs.offline.builder.OfflinePlanningRequestBuilder.build(window, prediction, spec) -> PlanningRequest`

No new replay path constructs `SchedulingRequest`, `ForecastPressure`, or legacy policy objects as its authority.

**Single evaluator**

Formal realized evaluator is now:

- `rs.offline.evaluation.OfflineEvaluator.evaluate(plan, truth, spec) -> PlanEvaluation`

Legacy `replay_and_audit_logical_plan()` remains compatibility-only.

**Truth / hint**

- planning sees `P0 actual + P1 actual + P2 hint`
- evaluation sees `ExecutionTruth`
- `ReplayEngine.execute` now carries both explicitly and keeps them separated

**Oracle**

- formal wrapper added at `rs.offline.oracle.cp_sat`
- environment lacks OR-Tools, so current result is fail-closed `UNSUPPORTED`

**Simulation**

- formal namespace created
- Phase A intentionally blocks unresolved execution semantics instead of inventing a second runtime model

**Why overall status is blocked**

Phase B requires:

- M123 integration genuinely READY
- offline/online plan parity
- materialization parity
- execution semantics parity
- solver-backed exact-local/exact-joint coverage

Those conditions are outside this branch’s ownership and are not faked here.
