# Comparison Configs

这个目录现在分成两层：

## 1. 当前推荐 public 入口

优先使用：

- `natural_256x128_4gpu.yaml`

它是当前主线推荐配置，采用收窄后的 public surface：

- `runtime.line`
- `runtime.output_mode`
- `strategies[].name`

推荐配置不再直接暴露这些内部行为开关：

- `execution_mode`
- `control_mode`
- `p2_hint_mode`
- `calibrated_p2`
- `bucket_rows`
- `heartbeat_enabled`
- `per_wave_timing_enabled`
- `replay_trace_enabled`
- `capture_enabled`

当前推荐含义：

- `runtime.line=phase_sync`
- `runtime.output_mode=paper`
- workload 固定为 `comparison_256x128_prompts.json`
- 策略固定为：
  - `disabled`
  - `birkhoff_phase_local`
  - `routersense_p0p1p2_hint`

## 2. legacy / ablation / historical configs

以下文件继续保留，但不再是当前默认入口：

- `tmp_comm_ramp_selected_4gpu.yaml`
- `tmp_comm_ramp_selected_bucket1024_4gpu.yaml`
- `tmp_comm_ramp_256x128_disabled.yaml`
- `paper_core_*`
- `paper_phase_local_*`
- `wire_slim_core_4gpu_64.yaml`
- `default.yaml`

这些配置仍有价值，但定位不同：

- 历史结果复现
- bucket / wire-size 消融
- 更早期的 phase-local 调试
- runner 兼容性保留

## Public line / output mode

当前 online public runtime 只对外暴露：

- runtime line:
  - `phase_sync`
  - `async_release`（当前仅声明，未实现）

- output mode:
  - `paper`
  - `debug_replay`

具体映射见：

- `docs/runtime_public_entrypoints.md`

## 当前建议

如果只是继续当前论文主线实验：

1. 默认先用 `natural_256x128_4gpu.yaml`
2. 只有做历史对照或消融时，才回到 `tmp_*` / `paper_*` 配置
