**Canonical builder**

- New authority: `rs.offline.builder.OfflinePlanningRequestBuilder.build(window, prediction, spec)`

**Legacy replay entrypoints**

- `rs.runtime.offline.replay_unified.ReplayEngine.execute`: `MIGRATE`
- `rs.runtime.offline.runner.build_scheduling_problem`: `THIN_WRAPPER`
- `experiments/run_offline_replay.py`: `THIN_WRAPPER`
- experiment-local `_build_problem*`: `REFERENCE_ONLY`

**Current closure**

`ReplayEngine.execute` now:

1. converts `ReplayWindow` -> `OfflineWindow`
2. converts replay hint -> formal `PredictionResult`
3. uses `OfflinePlanningRequestBuilder`
4. runs formal planner
5. evaluates via `OfflineEvaluator`
6. preserves legacy audit payload for compatibility
