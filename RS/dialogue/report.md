# Runtime Attribution Boundary Prep

## Goal
Refine the missing attribution points after the previous GPU run showed the extra 67-73 ms is not active P2P, compact preflight, P2P wait, or raw/core build.

## Changes
- Added selected MoE module forward pre/post hooks for selected layers only.
- Added best-effort expert module forward pre/post hooks under selected MoE layers.
- Added strict runtime-state fields for selected-layer timing records, expert-module timing records, and attribution boundary status.
- Exported those fields through rank prepared-plan summary.
- Added CPU tests proving strict runtime state can record measured selected/expert boundaries and can explicitly export expert boundary unavailable without fake expert compute.

## Measurement Semantics
- No CUDA synchronize, tensor CPU copy, item(), tolist(), or profiler was added.
- None layers still receive no RouterSense wrapper or attribution hook.
- Expert timing is only measured if a stable expert submodule is found; otherwise it is exported as unavailable.

## Current Blocker
GPU is currently unavailable in this shell (`nvidia-smi`: No devices were found). The next run must verify the new fields in real rank summaries.

## Next GPU Command
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. \
python experiments/distributed/run_gpu_a2_strategy_compare.py \
  --config configs/official/gpu_runtime_attribution.yaml \
  --output-dir outputs/tuning/runtime_attribution_gpu2 \
  --world-size 4 \
  --selected-layers 0,1 \
  --warmup-iters 1 \
  --measure-iters 2 \
  --profile attribution_light \
  --preflight-mode compact \
  --strategies native routersense_b_core_independent_async routersense_u_core_zero_raw_async
```
