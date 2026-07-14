**Formal evaluator**

- `rs.offline.evaluation.OfflineEvaluator.evaluate(plan, truth, spec)`

**Legacy evaluators**

- `rs.runtime.offline.runner.replay_and_audit_logical_plan`: `THIN_WRAPPER`
- `rs.runtime.offline.prediction.evaluation.compare_prediction`: `REFERENCE_ONLY`
- experiment-local schedule comparison logic: `REFERENCE_ONLY`

**Evaluator semantics in Phase A**

- authoritative realized result now comes from `OfflineEvaluator`
- legacy plan-replay audit remains compatibility-only diagnostic
- bucketized flows are validated by aggregated `(phase, src, dst)` coverage, not by single unsplit matrix-cell identity
