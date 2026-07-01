# Fast Scheduler Tail Closure - 2026-07-02

## Context
Pulled latest repo and checked `rs_channel/ins`. No new instruction beyond the existing fast-scheduler task, so this pass focused on closing remaining result-schema gaps for the no-GPU path.

## Completed
- Extended `run_pairwise_analysis()` output with:
  - `greedy_latency_ms`
  - `fast_latency_ms`
  - `oracle_perfect_latency_ms`
  - `oracle_predicted_latency_ms`
  - `oracle_prediction_gap_pct`
- Extended summary payload with corresponding aggregate stats.
- Updated analysis helpers to expose the new fields:
  - `analysis/summarize_pairwise.py`
  - `analysis/compare_strategies.py`
  - `analysis/plot_prediction.py`
  - `analysis/plot_makespan.py`
- Updated tests to assert the new fast/oracle gap and latency fields.

## Validation
- `cd RS && pytest -q tests/test_poc_line1.py tests/test_structure_refactor.py`
  - result: `17 passed`
- `cd RS && python analysis/plot_prediction.py --help`
  - ok
- `cd RS && python analysis/plot_makespan.py --help`
  - ok

## Ready State
The no-GPU pipeline is now set up so that once GPU runs resume, batch results can be consumed directly for:
- fast vs greedy vs oracle quality
- oracle-perfect vs oracle-predicted gap
- scheduling latency comparisons
