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

## 当前使用方式

当前第一版 replay trace 只提供统计输入，不做复杂重排。建议流程：

1. online perf 运行时写出 `rank*_control_replay_trace.jsonl`
2. offline 脚本读取这些行
3. 聚合控制面规模、wave 数、task_ref 数、wire size

后续如果要做“同一 trace 下换策略重排”，应当复用这个 schema，而不是把完整 debug artifact 强行塞进热路径。
