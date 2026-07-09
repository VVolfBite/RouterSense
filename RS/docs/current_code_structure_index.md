# Current Code Structure Index

这个文档只描述当前 mainline，帮助后续 Codex 快速定位主线。

## Online runtime

### `src/rs/runtime/online/megatron_ep/host.py`

- 负责什么：外部 attach 入口、把 formal config 接到 runtime
- 不负责什么：调度算法、executor 实现
- 是否属于 perf hot path：否
- 是否可以慢：可以稍慢

### `src/rs/runtime/online/megatron_ep/lifecycle.py`

- 负责什么：P0/P1 生命周期主线、phase plan agreement 调用、transport 激活、prepared-plan 状态
- 不负责什么：离线分析、绘图、重型 debug dump
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
- 不负责什么：真正的通信执行
- 是否属于 perf hot path：是
- 是否可以慢：不可以

### `src/rs/runtime/online/megatron_ep/execution/`

- 负责什么：transport adapter、sync wave executor、execution audit
- 不负责什么：策略选择、全局 root 协商
- 是否属于 perf hot path：是
- 是否可以慢：不可以

### `src/rs/runtime/online/megatron_ep/observation/views.py`

- 负责什么：把 phase context / plan / replay trace 压成轻量 artifact 视图
- 不负责什么：调度决策
- 是否属于 perf hot path：间接属于
- 是否可以慢：不应该太慢

### `src/rs/runtime/online/megatron_ep/observation/`

- 负责什么：recorder、artifact recorder、observer、trace writer
- 不负责什么：调度策略和 executor
- 是否属于 perf hot path：只有轻量 recorder 路径属于
- 是否可以慢：debug/execution 可慢，perf 不可重

### `src/rs/runtime/online/megatron_ep/contracts.py`

- 负责什么：online runtime 的共享 config / contract
- 不负责什么：行为逻辑
- 是否属于 perf hot path：否
- 是否可以慢：可以

## Offline / replay

### `src/rs/runtime/offline/`

- 负责什么：离线 trace、预测、重放、理论分析
- 不负责什么：真实 Megatron 执行
- 是否属于 perf hot path：否
- 是否可以慢：可以

### `experiments/offline/replay_online_control_trace.py`

- 负责什么：读取轻量 control replay trace，统计控制面规模
- 不负责什么：真实 GPU benchmark、复杂策略重排
- 是否属于 perf hot path：否
- 是否可以慢：可以

### `experiments/offline/build_replay_fixture_from_control_trace.py`

- 负责什么：把一组 `rank*_control_replay_trace.jsonl` 聚合成 offline scheduling fixture
- 不负责什么：真实 GPU 执行、NCCL 等待模拟
- 是否属于 perf hot path：否
- 是否可以慢：可以

## Experiments

### `experiments/online/run_strategy_comparison.py`

- 负责什么：在线多策略对比入口、public runtime surface 到内部开关的映射、子配置生成、结果聚合
- 不负责什么：runtime 逻辑本体

### `experiments/online/support/runtime_presets.py`

- 负责什么：把 `runtime.line` / `runtime.output_mode` / public strategy name 映射成现有内部字段
- 不负责什么：真实调度执行和 runtime 行为逻辑

### `configs/comparison/natural_256x128_4gpu.yaml`

- 当前推荐 public 主线配置
- workload 为 `configs/workload/comparison_256x128_prompts.json`

### `configs/comparison/README.md`

- 当前 comparison config 的 public/legacy 入口说明

## Scheduling

### `src/rs/scheduling/`

- 当前真实 runtime 使用的策略入口
- 包含：
  - phase-local baseline
  - fast-path 可调用的 online policy contract
  - offline logical scheduler / reference

特别说明：

- `birkhoff_phase_local` 是当前 online 可执行的 phase-local 强 baseline
- `routersense_p0p1p2_hint` 是当前 prediction-aware runtime policy
- oracle / heavy joint scheduler 属于 offline 或 theoretical upper bound
- 不应该把 offline oracle / heavy joint scheduler 塞回 online perf hot path

## Replay trace 如何打开

推荐 public 模式下：

- `runtime.output_mode=paper` 默认关闭 replay trace
- `runtime.output_mode=debug_replay` 默认开启 replay trace

legacy/internal 配置下，也可以显式开启：

```yaml
observation:
  profile: perf
  replay_trace_enabled: true
```

输出位置：

```text
rank*_control_replay_trace.jsonl
```

离线解析：

```bash
PYTHONPATH=src python -m experiments.offline.replay_online_control_trace \
  --trace outputs/.../rank0_control_replay_trace.jsonl
```
