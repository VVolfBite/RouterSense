# 2026-07-01 POC-line1 Pairwise Oracle Perf Update

## Status

- Scope: P1 supplementary performance diagnosis and optimization for `run_pairwise_analysis`
- Code path: `RS/src/routesense/evaluation/poc_line1.py`, `RS/experiments/poc_line1/pairwise_scheduler.py`
- Artifact: `RS/artifacts/poc_line1/pairwise_oracle_report_smoke_v3/`
- Perf log: `RS/artifacts/poc_line1/pairwise_oracle_report_smoke_v3.perf.log`

## What changed

1. Added end-to-end `[perf]` timing with `flush=True` in:
   - `run_pairwise_analysis`
   - `pairwise_scheduler.py` main entry
2. Confirmed preprocessing is not the bottleneck:
   - `_group_records`: ~0.00s
   - `build_sample_layer_matrices`: ~0.00s
   - `_decode_predicted_topk`: ~0.02-0.03s
   - `build_predicted_traffic`: ~0.000-0.001s per pair
3. Confirmed the bottleneck is `pairwise_oracle` / CP-SAT solve time on some layer pairs.
4. Tightened CP-SAT search:
   - time horizon capped to greedy makespan upper bound
   - `max_time_in_seconds = 1.0`
   - `num_search_workers = 8`
   - fixed `random_seed = 0`

## Smoke result

Initial timed smoke before the horizon tightening:

- total `run_pairwise_analysis`: `86.17s`
- dominant cost: repeated `pairwise_oracle` calls, several saturating the 5s limit

Timed smoke after tightening:

- total `run_pairwise_analysis`: `16.22s`
- total CLI wall time: `16.29s`

Representative per-pair solve times after tightening:

- `L0->1`: perfect `1.493s`, predicted `0.718s`
- `L2->3`: perfect `1.013s`, predicted `1.014s`
- `L11->12`: perfect `1.014s`, predicted `1.012s`
- light pairs remain sub-`0.1s`

## Gate 2 smoke summary

- `pair_count = 15`
- `perfect_improvement_pct.mean = 35.1925`
- `predicted_improvement_pct.mean = 35.2445`
- `traffic_correlation.mean = 0.6638`
- `predicted_improvement_vs_traffic_correlation = 0.3607`
- `gate2_decision = PASS`

## Validation

- `cd RS && pytest -q tests/test_poc_line1.py`
  - `16 passed in 3.16s`
- Smoke command:

```bash
cd /root/autodl-tmp/RouterSense/RS
OMP_NUM_THREADS=1 python -u experiments/poc_line1/pairwise_scheduler.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_smoke_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_smoke_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_smoke_v2/gate_weights.pt \
  --placement round_robin \
  --output-dir artifacts/poc_line1/pairwise_oracle_report_smoke_v3
```

## Next step

- With the current profiling evidence, the remaining cost is solver-side, not preprocessing.
- The next meaningful step is to run batch500 with the tightened CP-SAT path and check whether total wall time is now acceptable for P3.
