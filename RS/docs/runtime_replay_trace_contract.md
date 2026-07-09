# RouterSense Runtime Replay Trace Contract

`ControlReplayTrace` 是一个轻量控制面重放记录，不保存 tensor payload，也不试图替代真实 GPU benchmark。

## 目标

它只服务于离线分析：

- 统计 all_gather / broadcast 的规模
- 统计 wave / bucket / task_ref 数量
- 分析不同策略下控制面对象规模
- 为后续 offline replay 提供统一输入

## 不做什么

它不能：

- 精确重现 NCCL 排队等待
- 精确模拟 GPU runtime latency
- 保存真实 tensor
- 替代真实多卡性能实验

## 单条 phase 记录的最小字段

每个 scheduled phase 一行 JSONL，字段最小化为：

- `run_id_digest`
- `layer_id`
- `layer_name`
- `phase`
- `global_rank`
- `local_rank`
- `ep_group_size`
- `policy_name`
- `bucket_rows`
- `per_rank_peer_bytes`
- `nonzero_edges`
- `nonzero_edge_count`
- `p2_hint_summary`
- `abstract_plan_summary`
- `timing_summary`
- `transport_summary`

其中：

- `p2_hint_summary` 只保留 hint 模式、digest 和 preferred edge/wave 计数
- `abstract_plan_summary` 只保留 `plan_hash`、`wave_count`、`task_ref_count`
- `timing_summary` 只保留 `all_gather/build_plan/broadcast`
- `transport_summary` 只保留 `planning_summary_tensor_len`、`abstract_plan_tensor_len`、`bucket_count`、`total_wave_count`、`total_byte_count`、`hint_match_rate`

## 默认行为

- 默认关闭
- 只有显式打开 `observation.replay_trace_enabled=true` 才落盘
- `perf` profile 允许开启，因为它仍是轻量结构
- 当前 public 配置面下：
  - `runtime.output_mode=paper` 默认关闭
  - `runtime.output_mode=debug_replay` 默认开启

## 当前使用方式

当前第一版 replay trace 只提供统计输入，不做复杂重排。建议流程：

1. online perf 运行时写出 `rank*_control_replay_trace.jsonl`
2. offline 脚本读取这些行
3. 聚合控制面规模、wave 数、task_ref 数、wire size

后续如果要做“同一 trace 下换策略重排”，应当复用这个 schema，而不是把完整 debug artifact 强行塞进热路径。

## 最小使用示例

用 public 配置面时：

- `runtime.output_mode=paper` 不写 replay trace
- `runtime.output_mode=debug_replay` 会写 replay trace

如果使用 legacy/internal 配置，也可以手动打开：

```yaml
observation:
  profile: perf
  replay_trace_enabled: true
```

在线运行后，输出文件位于每个 rank 的 artifact 目录下：

```text
rank0_control_replay_trace.jsonl
rank1_control_replay_trace.jsonl
...
```

离线解析：

```bash
PYTHONPATH=src python -m experiments.offline.replay_online_control_trace \
  --trace outputs/.../rank0_control_replay_trace.jsonl
```

当前 parser 会输出：

- total phase 数
- all_gather / broadcast 调用数
- summary / plan element 总量
- task_ref 总量
- wave / bucket / nonzero_edge / total_byte 的平均值与最大值
- per_policy breakdown
- per_phase breakdown

## 当前还能做的最小桥接

现在可以把一组 `rank*_control_replay_trace.jsonl` 直接桥接成 offline scheduling fixture：

```bash
PYTHONPATH=src python -m experiments.offline.build_replay_fixture_from_control_trace \
  --trace-dir outputs/.../per_strategy/disabled/rep0 \
  --policy disabled \
  --output-dir outputs/.../replay_fixture_bundle
```

输出：

- `replay_fixture_bundle_summary.json`
- `replay_fixture_bundle.json`
- `fixtures/replay_layer_<id>.json`

这些 fixture 可以直接接到现有 offline scheduling / validation 工具，
用于下一轮 trace-driven replay，而不需要重新采集 tensor payload。
