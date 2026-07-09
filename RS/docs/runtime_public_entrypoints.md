# Runtime Public Entrypoints

这个文档只描述当前推荐给实验入口使用的 online 公共配置面。

## Online Runtime Lines

当前对外只保留两条 runtime line：

- `phase_sync`
  - 当前唯一已实现、已验证的 online 主线
  - 表示 phase barrier 下的计划生成、计划下发和 wave 执行
  - 当前 `birkhoff_phase_local` 和 `routersense_p0p1p2_hint` 都运行在这条线上

- `async_release`
  - 论文目标里的未来主线
  - 表示不依赖强 barrier、允许 rank 提前 release 的版本
  - 当前只有 shadow-only skeleton
  - 如果用户配置 `runtime.line=async_release`，入口必须直接报错：
    `async_release runtime_line has a shadow-only skeleton but no online executor integration yet`
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

而不是直接描述 runtime 内部行为细节。

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
- dispatch matrix 的全局构造必须走 tensorized gather，不能在 predictor / prepared-plan 路径里使用 Python object collective

## 与论文证据链的关系

- `phase_sync`
  - 当前真实可执行的 online 保守线
  - 用于证明 runtime 可复现、可审计、可 replay
- `async_release`
  - 当前只表达未来联合 release 语义
  - 还不能拿来声称 full joint online 已经实现

离线证据入口：

- `experiments/offline/build_replay_fixture_from_control_trace.py`
- `experiments/offline/run_real_trace_evidence_suite.py`

它们用于说明：

- 多 phase joint scheduling 是否有空间
- oracle / heuristic cross-layer prediction 是否有潜在收益
