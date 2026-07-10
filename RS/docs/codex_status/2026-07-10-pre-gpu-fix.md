# 2026-07-10 Pre-GPU Fix

## Scope

这轮只修 pre-GPU trace 前的阻断和口径问题，不跑 GPU，不跑大 benchmark。

修复目标：

1. async_release 不再伪装 real collectives 已执行
2. expert-to-traffic reconstruction 改为按 layer/source_rank 聚合完整 world matrix
3. expert_to_rank 映射严格要求 EP-local rank index
4. `fate_style_history` cold-start 不再用当前 sample fit 自己
5. calibration summary 不再把单条 `traffic_error_after_calibration` 当全局指标

## Result

- `async_release` 现在即使 flags 全开，也会因为 `real_collective_executor_implemented=false` 强制 fallback `phase_sync`
- `run_expert_to_traffic_reconstruction.py` 现在先 merge `source_expert_counts`，再做 O1/O2/O3/O4
- `source_expert_counts_to_traffic_matrix()` 现在：
  - 缺失 `expert_to_rank` 直接报错
  - `expert_to_rank` 超出 `[0, world_size)` 直接报错
  - 列索引必须是 EP-local rank
- prediction replay summary 只把 mean/median calibration error 当总体指标
- `fate_style_history` first layer cold-start 改成 `copy_current_dispatch` fallback，`used_current_sample_for_fit=false`

## GPU Next Step

下一轮只做：

1. collect expert route trace
2. verify `source_expert_counts` non-empty
3. run `run_expert_to_traffic_reconstruction.py`
4. compare O1/O2/O3/O4
5. only then decide whether to implement real gate replay predictor

## Hard Truths

- `gpu_not_run=true`
- `async_release_real_collectives_not_validated=true`
- `faithful_fate_not_validated=true`
