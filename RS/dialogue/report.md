# M-RUNTIME-TIMELINE-CPU Report

## Goal
Prepare perf-safe runtime timeline instrumentation and static evidence for the next 4GPU run. This round does not run CUDA/NCCL and does not change scheduling algorithms.

## Starting SHA
`8bc5acfea09ce2665c150fff314a237025df3188`

## What Changed
- Added explicit token count contract fields: `total_token_slots`, `valid_token_count`, `padding_token_count`, and `token_count_status`.
- Kept deprecated `padded_token_count` only as a compatibility field with `padded_token_count_unit=padding_token_count`.
- Added `RuntimePhaseTimeline` schema for per-rank, per-selected-layer, per-phase timestamps and derived intervals.
- Added async release timing fields for submit queue, submit span, request wait, active transport sum, and active transport critical path.
- Added task granularity and rank imbalance summary helpers.
- Added `configs/official/gpu_runtime_timeline.yaml` with `profile=timeline_light`, selected layers `0,1`, 8x16 workload, compact preflight, and B-core/U-zero strategies.
- Added static audit evidence for likely sources of the 80-90 ms selected communication window.

## Timing Semantics
`active_transport_sum_us` is batch submit plus work wait wall time accumulated by the async executor. `active_transport_critical_path_us` is first request submitted to all requests completed for that phase. The selected communication window is not pure network busy time; it can include phase gaps and compute/control gaps between P0 and P1.

## Static Audit Summary
Confirmed risks include release batches submitted and waited batch-by-batch, P0 matrix gather/plan agreement collectives before execution, preflight collectives in non-local modes, and old span naming that can be misread as network busy time. These are not optimized in this round.

## Tests
- `python -m compileall src experiments tests`
- `PYTHONPATH=src:. pytest -q tests/contract/test_runtime_timeline_cpu.py`
- `PYTHONPATH=src:. pytest -q tests/contract/test_gpu_child_config_and_a2_metrics.py tests/contract/test_runtime_measurement_semantics.py`

## Not Run
No GPU, NCCL, C2, first bring-up, seven-strategy A2, target lifecycle matrix, large workload sweep, profiler, or full pytest was run.

## Next GPU Command
```bash
python experiments/distributed/run_gpu_a2_strategy_compare.py \
  --config configs/official/gpu_runtime_timeline.yaml \
  --output-dir outputs/tuning/runtime_timeline_gpu \
  --world-size 4 \
  --selected-layers 0,1 \
  --warmup-iters 1 \
  --measure-iters 1 \
  --profile timeline_light \
  --preflight-mode compact \
  --strategies routersense_b_core_independent_async routersense_u_core_zero_raw_async
```

## Final Status
`RUNTIME_TIMELINE_CPU_READY`
