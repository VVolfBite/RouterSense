**Planning-visible**

- `OfflinePlanningRequestBuilder`
  - `window.p0_actual`
  - `window.p1_actual`
  - `prediction.hint.target_dispatch_rows`

**Evaluation-visible**

- `build_execution_truth`
  - `window.p0_actual`
  - `window.p1_actual`
  - `window.p2_actual`

**Compatibility replay path**

- `ReplayEngine.execute`
  - formal planning path uses hint only
  - formal evaluation path uses `ExecutionTruth`
  - legacy `MultiPhaseSchedulingProblem` remains compatibility audit only

**No truth leak**

- changing `p2_actual` does not change formal `PlanningRequest.semantic_digest`
- changing hint changes formal `PlanningRequest.semantic_digest`
