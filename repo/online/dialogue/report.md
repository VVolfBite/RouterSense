# Shadow Path Retire

## Goal
Retire duplicate legacy shadow work from the real async B-core and U-zero hot path while keeping lightweight observation finalize and preserving an explicit legacy shadow mode.

## Actual call chain
- Host wrapper still calls `before_token_dispatch -> facade.dispatch -> after_token_dispatch -> on_dispatch`.
- Host wrapper still calls `before_token_combine -> facade.combine -> after_token_combine -> on_combine`.
- Real async execution happens in `before_token_dispatch` and `before_token_combine`.
- Historical duplicate shadow work previously lived in `on_dispatch` and `on_combine`.

## This round
- Added explicit hook execution mode selection inside lifecycle.
- Current async real execution now uses `REAL_EXECUTION_WITH_OBSERVATION`.
- In that mode, `on_dispatch` and `on_combine` only run lightweight finalize logic.
- Legacy shadow logic remains available through explicit legacy scheduler modes.
- Added formal counters for real execution, shadow execution, observation finalize, and shadow agreement/build/control.
- Added DTOH callsite aggregation fields and split P0 matrix gather into local prepare / collective / DTOH decode timing stages.
- Added separate timing export support for `on_dispatch` and `on_combine`.

## Contracts now expected for B-core and U-zero
- `real_p0_execution_count == expected selected P0 count`
- `real_p1_execution_count == expected selected P1 count`
- `shadow_dispatch_execution_count == 0`
- `shadow_combine_execution_count == 0`
- `shadow_policy_agreement_count == 0`
- `shadow_plan_build_count == 0`
- `shadow_control_collective_count == 0`
- `observation_finalize_dispatch_count == expected selected dispatch count`
- `observation_finalize_combine_count == expected selected combine count`

## Not done in this CPU round
- No GPU rerun
- No performance claim
- No transport/task merge changes
- No algorithm changes

## Next GPU command
```bash
cd /root/autodl-tmp/RouterSense/RS
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. \
python experiments/distributed/run_gpu_a2_strategy_compare.py \
  --config configs/official/gpu_shadow_retire_check.yaml \
  --output-dir outputs/tuning/gpu_shadow_retire_check \
  --world-size 4 \
  --selected-layers 0,1 \
  --warmup-iters 1 \
  --measure-iters 2 \
  --profile attribution_light \
  --preflight-mode compact \
  --strategies native routersense_b_core_independent_async routersense_u_core_zero_raw_async
```
