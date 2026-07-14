**Formal prediction surface**

- `PredictionRegistry`
- `Predictor.predict`
- `PredictionEvaluator`

**Offline usage in this branch**

- `OfflinePlanningRequestBuilder` consumes `PredictionResult`
- `ReplayEngine.execute` now converts replay hints into `PredictionResult`
- `prediction_digest()` added as the formal offline record digest input
- `rs.offline.rollout.run_prediction_rollout()` adds a formal roll-forward offline prediction path with explicit split and cold-start semantics

**Still legacy / reference**

- `runtime/offline/prediction/evaluation.py`: historical rollout utility, not authoritative
- experiment-local predictor studies: `REFERENCE_ONLY`

**Phase B**

- split train/validation/test rollout
- offline/online prediction parity evidence
- scheduling-regret bundle on top of realized `OfflineEvaluator`
