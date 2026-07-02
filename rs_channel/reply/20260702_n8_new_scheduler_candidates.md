# N=8 New Scheduler Candidates Status

Date: 2026-07-02 UTC

## Implemented

Added the following new schedulers:

- `birkhoff_exhaustive`
- `ejection_chain_tabu`
- `lns_cp_repair`
- `decomposed`
- `quantized_decomposed`

Integrated them into:

- `RS/src/routesense/scheduler/fast.py`
- `RS/src/routesense/scheduler/strategies.py`
- `RS/src/routesense/scheduler/__init__.py`
- `RS/src/routesense/evaluation/analysis.py`
- `RS/src/routesense/evaluation/__init__.py`

Also added:

- `RS/experiments/poc_line1/exp_pairwise_candidate_compare.py`

## Validation

- `python -m pytest RS/tests -q`
  - `43 passed, 1 warning`

## Small-sample N=8 smoke

Artifact:
- `RS/artifacts/poc_line1/candidate_compare_sample8/table.md`

Result snapshot:

| algorithm | mean_improvement_pct | mean_latency_ms |
|---|---:|---:|
| birkhoff | 22.95 | 1.03 |
| ibbr | 24.47 | 6.86 |
| ejection_chain_tabu | 27.03 | 11.14 |
| lns_cp_repair | 27.03 | 11.33 |
| decomposed | 7.51 | 47.09 |
| quantized_decomposed | 6.35 | 21.85 |

## Readout

- `ejection_chain_tabu` and `lns_cp_repair` do improve over `birkhoff` and `ibbr`, which confirms that moving outside the strict phase-optimal subspace helps.
- Current gap to the target remains:
  - improvement is still below the desired `30%+`
  - latency is still above the desired `<5ms`
- `decomposed` and `quantized_decomposed` are not yet competitive in this first implementation.

## Interpretation

This is consistent with the earlier diagnosis:
- the useful direction is indeed cross-phase search with controlled degradation;
- but the current implementations still spend too much time in repeated global schedule evaluation;
- the next optimization step should focus on reducing evaluation cost per move, not just adding more move types.
