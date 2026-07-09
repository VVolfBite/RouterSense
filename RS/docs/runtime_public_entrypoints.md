# Runtime Public Entrypoints

这个文档只描述当前推荐给实验入口使用的 online 公共配置面。

## Online Runtime Lines

当前对外只保留两条 runtime line：

- `phase_sync`
  - 当前唯一已实现、已验证的 online 主线
  - 语义上代表 phase barrier 下的计划生成、计划下发和 wave 执行
  - 当前 `birkhoff_phase_local` 和 `routersense_p0p1p2_hint` 都运行在这条线上

- `async_release`
  - 论文目标里的未来主线
  - 代表不依赖全局强 barrier、允许 rank 提前 release 的版本
  - 当前尚未实现
  - 如果用户配置 `runtime.line=async_release`，入口必须直接报错
  - 不能偷偷 fallback 到 `phase_sync`

## Online Output Modes

当前对外只保留两种 output mode：

- `paper`
  - 面向论文性能对比
  - 轻量
  - 默认关闭 replay trace、tensor capture、heartbeat 和 per-wave timing
  - 内部映射到 `observation.profile=perf`

- `debug_replay`
  - 面向 debug 和 offline replay
  - 允许保留更多 lightweight artifact
  - 默认打开 replay trace
  - capture 仍然默认关闭；如果需要 tensor capture，必须额外显式给 selector
  - 内部映射到当前 debug profile

## 当前推荐配置应该描述什么

推荐配置应该描述：

- 模型路径
- EP size
- workload
- runtime line
- output mode
- 要对比的策略

而不是直接描述 runtime 内部细节。

不应再在推荐配置里直接暴露这些 legacy internal knobs：

- `execution_mode`
- `control_mode`
- `p2_hint_mode`
- `calibrated_p2`
- `bucket_rows`
- `heartbeat_enabled`
- `per_wave_timing_enabled`
- `replay_trace_enabled`
- `capture_enabled`

这些仍然可以存在于 legacy / ablation 配置里，但不是当前 mainline public surface。

## 当前 phase_sync 的内部映射

### `disabled`

- internal run kind: `online_observe`
- internal execution mode: `native_passthrough`
- internal control mode: `none`

### `birkhoff_phase_local`

- internal run kind: `online_policy_correctness`
- internal execution mode: `phase_sync_wave`
- internal control mode: `sync_before_phase`
- internal P2 mode: `none`

### `routersense_p0p1p2_hint`

- internal run kind: `online_policy_correctness`
- internal execution mode: `multiphase_pending_window`
- internal control mode: `sync_before_phase`
- internal P2 mode: `calibrated_artifact`

注意：

- 当前 online RouterSense 仍然是 prediction-aware phase-local runtime policy
- 它不是完整的 online multiphase live pending queue executor
- fast path 仍然保留 root-authoritative agreement 和 wave-level plan

## Offline 为什么可以更复杂

`runtime/offline/` 和 `experiments/offline/` 可以保留更多 replay/debug/analysis 入口，因为：

- offline 不在热路径
- offline 可以慢
- offline 可以承接理论分析、oracle、trace replay 和 wire-size study

online 入口则必须收窄，避免历史实验开关继续污染主线。
