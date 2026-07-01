## Scope

- Phase: POC-line1 scheduler pipeline refactor
- Model default remains `full_duplex`
- Goal: remove weak candidates from the analysis pipeline and add two cross-phase candidates:
  - `phase_aware_greedy`
  - `lagrangian`

## Blocking Fixes Confirmed

1. `oracle.py`
   - `half_duplex` barrier now uses receiver -> sender dependency
   - `expert_compute_delay` is applied consistently

2. Default model
   - default `model=` is now `full_duplex` across scheduler/evaluation entrypoints

3. `analysis.py`
   - summary now includes:
     - `oracle_perfect_solver_statuses`
     - `oracle_predicted_solver_statuses`

## Pipeline Changes

Removed from `run_pairwise_analysis` summary pipeline:
- `lookahead_lpt`
- `cp_lpt`
- `cp_local_swap`

Added to `run_pairwise_analysis` summary pipeline:
- `phase_aware_greedy`
- `lagrangian`

Retained core methods:
- `greedy`
- `birkhoff`
- `iterated_greedy`
- `fast_pairwise`
- `oracle_perfect`
- `oracle_predicted`

## fast_schedule_pairwise

`fast_schedule_pairwise` is now a reduced best-of set:
- `birkhoff`
- `lagrangian`
- `iterated_greedy`

Selection rule unchanged:
- prefer candidates with `solve_time_ms <= 5.0`
- choose smallest makespan, then smaller solve time

## New Methods

### phase_aware_greedy

- LPT-style heuristic with downstream receive-pressure look-ahead
- intended as a very fast cross-phase baseline

### lagrangian

- lightweight Lagrangian-relaxation style coordination over the 3 phases
- updates multipliers for receiver->sender barrier violations
- intended to sit between `birkhoff` and `iterated_greedy`

## Validation

Command:
- `cd RS && PYTHONPATH=src python -m pytest tests/ -x -q`

Result:
- `35 passed, 1 warning`

Additional smoke check:
- `run_pairwise_analysis(...)` now emits:
  - `phase_aware_greedy_improvement_pct`
  - `lagrangian_improvement_pct`
  - `oracle_perfect_solver_statuses`
  - `oracle_predicted_solver_statuses`

## Files Changed

- `RS/src/routesense/scheduler/fast.py`
- `RS/src/routesense/scheduler/__init__.py`
- `RS/src/routesense/evaluation/analysis.py`
- `RS/src/routesense/evaluation/__init__.py`
- `RS/src/routesense/evaluation/poc_line1.py`
- `RS/tests/test_poc_line1.py`

## Current Status

- Refactor is code-complete and test-clean.
- No new GPU batch run was started in this step.
- Next useful run is a refreshed `full_duplex` batch500 compare to place:
  - `phase_aware_greedy`
  - `lagrangian`
  - `birkhoff`
  - `iterated_greedy`
  - `oracle_perfect`
  - `oracle_predicted`
  on one table.
