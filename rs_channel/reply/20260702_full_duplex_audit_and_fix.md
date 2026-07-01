## Phase Status

- Scope: `full_duplex` only for mainline POC-line1 scheduling comparison.
- Active conclusion remains valid:
  - `oracle_perfect` materially outperforms `fast_birkhoff`.
  - This preserves the line1 thesis: cross-layer joint scheduling adds value beyond per-layer independent Birkhoff scheduling.

## Code Fixes Applied

1. Default scheduling model switched from `half_duplex` to `full_duplex` across:
   - `src/routesense/scheduler/greedy.py`
   - `src/routesense/scheduler/oracle.py`
   - `src/routesense/scheduler/fast.py`
   - `src/routesense/evaluation/analysis.py`

2. `oracle.py` phase barrier bug fixed for `half_duplex` baseline:
   - barrier now uses receiver -> sender dependency
   - `expert_compute_delay` is now applied consistently

3. `analysis.py` now records:
   - `oracle_perfect_solver_statuses`
   - `oracle_predicted_solver_statuses`

4. `tests/test_package_source_only.py` portability fix:
   - preserves existing `PYTHONPATH` instead of overwriting it

5. Added explanatory comments in `fast.py` for:
   - Birkhoff round ordering semantics
   - critical-path weight accounting under full-duplex ports

## Validation

- Command:
  - `cd RS && PYTHONPATH=src python -m pytest tests/ -x -q`
- Result:
  - `35 passed, 1 warning`

## Key Audit Result

Artifact:
- `RS/artifacts/poc_line1/pairwise_model_compare_full_duplex_audit_v1/full_duplex_summary.json`

Important fields:
- `oracle_perfect_solver_statuses = {"OPTIMAL": 240}`
- `oracle_predicted_solver_statuses = {"OPTIMAL": 240}`
- `perfect_improvement_pct.mean = 41.4371`
- `predicted_improvement_pct.mean = 41.5603`
- `birkhoff_improvement_pct.mean = 20.4446`
- `iterated_greedy_improvement_pct.mean = 32.0317`
- `lookahead_lpt_improvement_pct.mean = -9.9087`
- `traffic_correlation.mean = 0.7465`

Interpretation:
- The earlier anomaly `oracle_predicted > oracle_perfect` is not a modeling win for prediction.
- In this audit slice, both solver-status counters are fully `OPTIMAL`, so the remaining tiny gap is numerical / instance-level noise rather than timeout-driven FEASIBLE mismatch.
- The main structural result is stable:
  - `oracle_perfect` beats `fast_birkhoff` by about `21.0` percentage points on this audit slice.

## Existing Batch500 Reference

Artifact:
- `RS/artifacts/poc_line1/pairwise_model_compare_batch500_v1/full_duplex_summary.json`

Previously observed main numbers:
- `birkhoff_improvement_pct.mean = 19.7858`
- `iterated_greedy_improvement_pct.mean = 29.3963`
- `oracle_perfect_improvement_pct.mean ≈ 40.0007`
- `oracle_predicted_improvement_pct.mean ≈ 40.2327`

Use this file for the large-batch headline until a refreshed full-duplex-only rerun is requested.

## Current Read

- Main model should stay `full_duplex`.
- `incast_only` is too loose and should remain only as a control / lower-bound variant.
- The core comparison for the paper line is:
  - `greedy`
  - `fast_birkhoff`
  - `iterated_greedy`
  - `oracle_perfect`
  - `oracle_predicted`

## Next Recommended Step

- Run a refreshed `full_duplex`-only batch500 compare with solver-status logging retained, then inspect:
  - `oracle_perfect` vs `fast_birkhoff`
  - `oracle_perfect` vs `iterated_greedy`
  - whether `lookahead_lpt` should be removed from the candidate set
