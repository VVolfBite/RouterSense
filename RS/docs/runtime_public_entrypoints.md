# Runtime Public Entrypoints

这个文档只描述当前推荐给实验入口使用的 online 公共配置面。

## Online Runtime Lines

当前对外只保留两条 runtime line：

- `phase_sync`
  - 当前保守同步线
  - 表示 phase barrier 下的计划生成、计划下发和 wave 执行
  - 当前 `birkhoff_phase_local` 和 `routersense_p0p1p2_hint` 都运行在这条线上

- `async_release`
  - 当前正式 async P2P 线
  - 表示使用 `joint_window_async_p2p` 执行模式与真实 `batch_isend_irecv`
  - 已通过 B2 lifecycle gate
  - 当前仍需后续 GPU 会话完成 C2 correctness 和 A2 performance
  - 不允许偷偷 fallback 到 `phase_sync`

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

### `birkhoff_bucket_phase_local`

- internal run kind: `online_policy_correctness`
- internal policy alias: `birkhoff_phase_local`
- `phase_sync` line -> `phase_sync_wave`
- `async_release` line -> `joint_window_async_p2p`
- internal control mode: `sync_before_phase`
- internal P2 mode: `none`

### `routersense_joint_zero_hint_async_p2p`

- internal policy: `routersense_p0p1p2_hint`
- internal execution mode: `joint_window_async_p2p`
- internal control mode: `sync_before_phase`
- internal P2 mode: `none`

### `routersense_joint_predicted_async_p2p`

- internal run kind: `online_policy_correctness`
- internal policy: `routersense_p0p1p2_hint`
- internal execution mode: `joint_window_async_p2p`
- internal control mode: `sync_before_phase`
- internal P2 mode: `calibrated_artifact`

注意：

- 当前 online RouterSense 仍然是 prediction-aware phase-local runtime policy
- 它不是完整的 online multiphase live pending queue executor
- fast path 仍然保留 root-authoritative agreement 和 wave-level plan
- dispatch matrix 的全局构造必须走 tensorized gather，不能在 predictor / prepared-plan 路径里使用 Python object collective

## 与论文证据链的关系

- `phase_sync`
  - 当前同步参考线
  - 用于 C2/A2 的 reference 和 backend 对照
- `async_release`
  - 当前真实 async P2P 线
  - B2 已确认信息链和 transport 接入
  - C2/A2 仍待完成 correctness/performance

离线证据入口：

- `experiments/offline/build_replay_fixture_from_control_trace.py`
- `experiments/offline/run_real_trace_evidence_suite.py`

它们用于说明：

- 多 phase joint scheduling 是否有空间
- oracle / heuristic cross-layer prediction 是否有潜在收益
