# 2026-07-01 POC-line1 CP-SAT 200ms Update

## Scope

- Reduced CP-SAT per-call time budget from `1.0s` to `0.2s`
- Kept the indexing optimizations from the previous step
- Re-ran smoke
- Restarted `batch500` under the new 200ms solver budget

## Code change

File:

- `RS/src/routesense/evaluation/poc_line1.py`

Change:

- `solver.parameters.max_time_in_seconds = 0.2`

`FEASIBLE` remains accepted as a valid solver status.

## Validation

- `cd RS && pytest -q tests/test_poc_line1.py`
  - `16 passed in 3.31s`

## Smoke result

Artifact:

- `RS/artifacts/poc_line1/pairwise_oracle_report_smoke_v5/`
- perf log: `RS/artifacts/poc_line1/pairwise_oracle_report_smoke_v5.perf.log`

Summary:

- total wall time: `5.63s`
- total `run_pairwise_analysis`: `5.56s`
- smoke target `< 10s` met
- many CP-SAT calls now return around `0.209s` / `0.210s`
- Gate 2 remains `PASS`

Key summary values:

- `perfect_improvement_pct.mean = 35.1402`
- `predicted_improvement_pct.mean = 35.2445`
- `traffic_correlation.mean = 0.6638`
- `gate2_decision = PASS`

## Batch500 restart

Active run:

- output dir: `RS/artifacts/poc_line1/pairwise_oracle_report_batch500_v3/`
- perf log: `RS/artifacts/poc_line1/pairwise_oracle_report_batch500_v3.perf.log`

Observed early perf:

- `trace loaded`: `8.26s`
- `hidden states loaded`: `1.33s`
- `gate weights loaded`: `3.09s`
- `_group_records`: `0.35s`
- `build_sample_layer_matrices`: `0.31s`
- `_decode_predicted_topk`: `1.17s`
- most solver calls are now either:
  - `~0.21s` with `status=FEASIBLE`
  - much lower when `OPTIMAL`

This confirms the 200ms budget is active and the batch run is now bounded by the intended POC-time budget rather than the previous 1s cap.
