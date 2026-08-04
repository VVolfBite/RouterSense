Executed in Phase A:

- `tests/offline/test_offline_core.py`
- `tests/solver/test_offline_oracle_cp_sat.py`
- `tests/contract/test_truth_hint_isolation.py`
- `tests/offline/test_prediction_planning_parity.py`
- `python -m compileall RS/src RS/tests`
- `experiments/run_offline_replay.py --config RS/tests/fixtures/configs/minimal_offline.yaml`

Covered functions:

- `OfflinePlanningRequestBuilder.build`
- `build_evaluation_task_set`
- `build_execution_truth`
- `OfflineEvaluator.evaluate`
- `solve_cp_sat` unsupported path
- `ReplayEngine.execute` compatibility path via smoke

Not yet covered in Phase A:

- OR-Tools-backed optimal solve
- materialized-plan parity
- online functional execution parity
