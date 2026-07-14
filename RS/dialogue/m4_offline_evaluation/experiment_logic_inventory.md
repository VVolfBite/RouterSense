Experiment-local logic still present:

- `experiments/offline/run_prediction_oracle_baseline_closure.py`
- `experiments/offline/run_oracle_gap_replay.py`
- `experiments/offline/run_prediction_replay_suite.py`
- `experiments/offline/run_replay_fixture_policy_suite.py`

Classification:

- all remain `REFERENCE_ONLY` in Phase A
- they are not treated as the formal offline core
- formal offline core now lives under `rs.offline`

Deletion is deferred because those scripts still carry paper-era reporting and solver logic that needs extraction first.
