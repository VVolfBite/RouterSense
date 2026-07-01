# 2026-07-01 POC-line1 Pairwise Indexing Update

## Scope

- Implemented the requested algorithm-level indexing optimizations for `run_pairwise_analysis`
- Kept solver semantics unchanged
- Re-ran smoke and started `batch500` with the new indexing path

## Code changes

Files:

- `RS/src/routesense/evaluation/poc_line1.py`

Changes:

1. `build_sample_layer_matrices(..., grouped=None)`
   - now accepts a prebuilt `grouped` map
   - no longer recomputes `_group_records_by_sample_token_layer`
   - precomputes `tokens_by_sample` once instead of rescanning `records` per sample

2. `build_predicted_traffic(..., token_index=None)`
   - now accepts prebuilt `token_index[(sample_id, layer_id)] -> token_positions`
   - avoids O(n) scan over all grouped keys on every call
   - preserves fallback logic if no index is passed

3. `run_pairwise_analysis(...)`
   - prebuilds `token_index` once
   - passes `grouped=grouped` into `build_sample_layer_matrices`
   - passes `token_index=token_index` into `build_predicted_traffic`
   - reduces perf log volume so batch runs are not dominated by stdout I/O

## Validation

- `cd RS && pytest -q tests/test_poc_line1.py`
  - `16 passed in 3.25s`

- Smoke rerun:
  - artifact: `RS/artifacts/poc_line1/pairwise_oracle_report_smoke_v4/`
  - perf log: `RS/artifacts/poc_line1/pairwise_oracle_report_smoke_v4.perf.log`
  - total wall time: `16.21s`
  - no regression versus prior `16.29s`
  - Gate 2 remained `PASS`

## Batch500 status

Active output:

- `RS/artifacts/poc_line1/pairwise_oracle_report_batch500_v3/`
- `RS/artifacts/poc_line1/pairwise_oracle_report_batch500_v3.perf.log`

Observed early-stage perf after indexing fix:

- `trace loaded`: `7.16s`
- `hidden states loaded`: `1.06s`
- `gate weights loaded`: `2.87s`
- `_group_records`: `0.33s`
- `build_sample_layer_matrices`: `0.33s`
- `_decode_predicted_topk`: `1.28s`
- `build_predicted_traffic`: effectively `0.000s` per pair in sampled logs

This confirms the indexing fix worked: preprocessing is now cheap and stable even on the batch trace. Remaining wall time is solver-side.

## Note

`full_sequence_trace_batch500_v2` contains exactly `500` unique samples, but sample IDs are sparse strings such as `prompt-1006`, `prompt-9049`. Large numeric suffixes do not indicate extra sample count.
