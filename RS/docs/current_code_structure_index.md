# Current Code Structure Index

这个文档只描述当前 mainline，帮助后续 Codex 先看懂主线，再决定要不要深入某个目录。

## Online runtime

### `src/rs/runtime/online/megatron_ep/host.py`

- 负责什么：外部 attach 入口、formal config 接线
- 不负责什么：调度算法、executor 实现
- 是否属于 perf hot path：否
- 是否可以慢：可以稍慢

### `src/rs/runtime/online/megatron_ep/lifecycle.py`

- 负责什么：P0/P1 生命周期主线、phase plan agreement 调用、transport 激活、prepared-plan 状态
- 不负责什么：offline 分析、绘图、重型 dump
- 是否属于 perf hot path：是
- 是否可以慢：不可以

### `src/rs/runtime/online/megatron_ep/control/plan_agreement.py`

- 负责什么：planning summary gather、root plan build、abstract plan broadcast、local materialize
- 不负责什么：payload 执行
- 是否属于 perf hot path：是
- 是否可以慢：不可以

### `src/rs/runtime/online/megatron_ep/pending_window/adapter.py`

- 负责什么：prepared window / fast path 接口，连接当前 phase policy
- 不负责什么：executor、global P2 修复
- 是否属于 perf hot path：是
- 是否可以慢：不可以

### `src/rs/runtime/online/megatron_ep/pending_window/policy_adapter.py`

- 负责什么：prepared priority 到当前 phase policy 的适配
- 不负责什么：真实 transport 执行
- 是否属于 perf hot path：是
- 是否可以慢：不可以

### `src/rs/runtime/online/megatron_ep/execution/`

- 负责什么：transport adapter、sync wave executor、execution audit
- 不负责什么：策略选择、全局 root 协商
- 是否属于 perf hot path：是
- 是否可以慢：不可以

### `src/rs/runtime/online/megatron_ep/observation/`

- 负责什么：recorder、artifact 视图、replay trace writer
- 不负责什么：调度决策与 executor
- 是否属于 perf hot path：轻量 recorder 路径属于
- 是否可以慢：debug/execution 可以稍重，perf 不可重

### `src/rs/runtime/online/megatron_ep/async_release/`

- 负责什么：future async-release 的 shadow-only contract、state machine、validation skeleton
- 不负责什么：真实 online executor、真实通信 launch
- 是否属于 perf hot path：否
- 是否可以慢：可以

## Offline / replay

### `src/rs/runtime/offline/`

- 负责什么：offline trace、预测、replay、理论分析
- 不负责什么：真实 Megatron 执行
- 是否属于 perf hot path：否
- 是否可以慢：可以

### `experiments/offline/replay_online_control_trace.py`

- 负责什么：读取 lightweight control replay trace，统计 control-plane 规模
- 不负责什么：真实 GPU benchmark、复杂策略重排
- 是否属于 perf hot path：否
- 是否可以慢：可以

### `experiments/offline/build_replay_fixture_from_control_trace.py`

- 负责什么：把 `rank*_control_replay_trace.jsonl` 聚合成 offline scheduling fixture，并生成 fixture audit
- 不负责什么：真实 GPU 执行、NCCL 等待模拟
- 是否属于 perf hot path：否
- 是否可以慢：可以

### `experiments/offline/run_real_trace_evidence_suite.py`

- 负责什么：基于真实 replay fixture 生成三张论文证据表
- 不负责什么：真实 GPU benchmark、online 收益声明
- 是否属于 perf hot path：否
- 是否可以慢：可以

### `experiments/offline/run_transport_stress_replay.py`

- 负责什么：把真实 replay fixture 重放成 communication-only transport-stress 报告
- 不负责什么：真实 GPU benchmark、online runtime 执行
- 是否属于 perf hot path：否
- 是否可以慢：可以

## Experiments

### `experiments/online/run_strategy_comparison.py`

- 负责什么：在线多策略对比入口、public runtime surface 到内部字段的映射、子配置生成、结果聚合
- 不负责什么：runtime 逻辑本体

### `experiments/online/support/runtime_presets.py`

- 负责什么：把 `runtime.line` / `runtime.output_mode` / public strategy name 映射成现有内部字段
- 不负责什么：真实调度执行与 runtime 状态管理

### `configs/comparison/natural_256x128_4gpu.yaml`

- 当前推荐 public 主线配置
- workload 为 `configs/workload/comparison_256x128_prompts.json`

## Scheduling

### `src/rs/scheduling/`

- 当前真实 runtime 使用的策略入口
- 包含：
  - online 可调用的 phase-local policy
  - offline logical scheduler / reference
  - validation / replay contract

特别说明：

- `birkhoff_phase_local` 是当前 online 可执行的 phase-local 强 baseline
- `routersense_p0p1p2_hint` 是当前 prediction-aware runtime policy
- `B_birkhoff_wave` / `U_*` 属于 offline 或 theoretical upper bound
- 不应该把 offline oracle / heavy joint scheduler 塞回 online perf hot path

## 当前论文证据入口

- Claim 1：multi-phase joint scheduling space
  - `experiments/offline/run_real_trace_evidence_suite.py`
  - 表 B：`B_birkhoff_wave` vs `U_*`
- Claim 2：cross-layer prediction value
  - `experiments/offline/run_real_trace_evidence_suite.py`
  - 表 C：`zero_hint` / `copy_current_dispatch` / `perfect_trace` / `actual_trace`
- Claim 3：real reproducible online runtime
  - `runtime.line=phase_sync`
  - replay trace + audit
  - `async_release` 当前只有 skeleton
