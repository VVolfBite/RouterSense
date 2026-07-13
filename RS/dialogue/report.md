# M-RUNTIME-DIAG-GPU1 Temporary Report

## Goal

Collect the first 4GPU diagnostic run and split the 200+ ms selected window into preflight, hook path, transport, expert compute, inter-phase gap, and task fragmentation.

## Starting State

- starting commit: `235fa5a78652afbb0fc84da677e837118b3b414c`
- GPU run executed with 4 x RTX 4090D.
- Command used:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/distributed/run_gpu_a2_strategy_compare.py \
  --config configs/official/gpu_runtime_diag.yaml \
  --output-dir outputs/tuning/runtime_diag_gpu \
  --world-size 4 \
  --selected-layers 0,1 \
  --warmup-iters 1 \
  --measure-iters 1 \
  --profile timeline_light \
  --preflight-mode compact \
  --strategies routersense_b_core_independent_async routersense_u_core_zero_raw_async
```

## What Passed

- selected layers stayed at `[0,1]`.
- `none` layers stayed out of the heavy wrapper path.
- `selected_p0_hook_count` and `selected_p1_hook_count` were exact.
- compact preflight matched executor mode on every rank.
- raw-U build count stayed within the per-rank and all-rank upper bounds.
- tokenization stayed fixed at `8 x 16` with `total_token_slots=128`, `valid_token_count=72`, `padding_token_count=56`.

## Measured Structure

### B-core

- `total_forward_us`: `220791.015625`
- `dispatch_hook_path_us`: `79371.445`
- `combine_hook_path_us`: `28175.388`
- `dispatch_transport_us`: `5206.957`
- `return_transport_us`: `3965.395`
- `all_rank_transport_span_us`: `9172.352`
- `expert_compute_us`: `113244.183`
- `raw_u_build_us`: `8981.586`
- `preflight_us`: `118.811`
- `unattributed_us`: `201788.452`
- `task_count`: `72`
- `wave_count`: `18`
- `p2p_op_count`: `36`

### U-zero

- `total_forward_us`: `225332.001`
- `dispatch_hook_path_us`: `80635.741`
- `combine_hook_path_us`: `28846.536`
- `dispatch_transport_us`: `5252.688`
- `return_transport_us`: `2622.502`
- `all_rank_transport_span_us`: `7875.190`
- `expert_compute_us`: `115266.594`
- `raw_u_build_us`: `8686.665`
- `preflight_us`: `101.200`
- `unattributed_us`: `207928.576`
- `task_count`: `72`
- `wave_count`: `24`
- `p2p_op_count`: `36`

## Interpretation

- Active transport is still only single-digit milliseconds.
- The remaining selected-window time is dominated by hook path, expert compute, and a 9.8-14.7 ms P0-to-P1 gap.
- `unattributed_us` is still large, so the current timeline is useful for attribution but not a closed root-cause proof.
- Rank 3 is not the only critical rank in this run; the slowest rank shifts by strategy and metric.

## Tests

- `python -m compileall src experiments tests`: passed.
- `PYTHONPATH=src:. pytest -q tests/contract/test_runtime_diag_cpu.py tests/contract/test_runtime_timeline_cpu.py tests/contract/test_gpu_child_config_and_a2_metrics.py`: 33 passed.

## Not Run

- No full C2.
- No seven-strategy A2.
- No target lifecycle matrix.
- No profiler trace.

## Temporary Conclusion

The first 4GPU diagnostic run is complete and the communication window is now split into measurable pieces. The next step is to use the existing timeline to decide whether the remaining gap is mainly expert compute, hook overhead, or control/synchronization work.
