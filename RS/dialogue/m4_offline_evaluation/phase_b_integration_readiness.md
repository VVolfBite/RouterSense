**Scope**

This note is the Phase B handoff boundary for M4.

Phase A is complete on this branch. Phase B remains blocked on an actually READY `M123` integration baseline.

**What Phase B must consume**

From `M123` integration, M4 needs stable access to:

1. online sync `PlanningRequest`
2. online sync `WindowPlan`
3. online `MaterializedPlan`
4. functional Gloo execution summaries / completed-task sets
5. final M2 execution digests

**Required parity levels**

1. Input parity:
   - offline `PlanningRequest.semantic_digest`
   - online sync `PlanningRequest.semantic_digest`

2. Plan parity:
   - offline `WindowPlan.semantic_digest`
   - online sync `WindowPlan.semantic_digest`

3. Materialization parity:
   - offline materialized digest
   - online sync materialized digest

4. Execution semantics parity:
   - offline expected completed tasks
   - Gloo functional completed tasks

**Current blockers**

1. `convergence/m123-integration` exists, but has not been certified READY as the formal Phase B baseline.
2. M2 materialization/execution contracts are not yet the authoritative review baseline for offline/online parity.
3. CP-SAT coverage remains `UNSUPPORTED` in the current environment because OR-Tools is unavailable.

**What M4 already provides for Phase B**

1. Formal offline builder:
   - `rs.offline.builder.OfflinePlanningRequestBuilder`
2. Formal execution truth:
   - `rs.offline.builder.build_execution_truth`
3. Formal realized evaluator:
   - `rs.offline.evaluation.OfflineEvaluator`
4. Formal comparison / aggregation helpers:
   - `rs.offline.analysis`
5. Formal fairness gate:
   - `rs.offline.fairness`
6. Formal prediction rollout:
   - `rs.offline.rollout`
7. Formal logical-plan simulation contract:
   - `rs.simulation`

**Phase B implementation entry**

When `M123_PARALLEL_INTEGRATION_READY` is available:

1. add online-to-offline parity adapters under `rs.offline`
2. ingest online sync request/plan/materialization digests
3. build parity records into `OfflineEvaluationBundle`
4. emit:
   - `input_parity.json`
   - `plan_parity.json`
   - `materialization_parity.json`
   - `execution_semantics_parity.json`

**Non-goals**

This document does not authorize modifying:

- M1 lifecycle / host / runtime state
- M1 control lane
- M2 plan publisher / executor internals
- M3 artifact writer / debug / measurement internals

Those remain owned by their respective tracks.
