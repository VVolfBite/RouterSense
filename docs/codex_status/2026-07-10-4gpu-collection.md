# 2026-07-10 4GPU Collection

## Scope

本轮按 `Run B -> Run C -> Run A` 顺序完成了单机 4GPU 真实采集。

- Run B: debug expert trace collection
- Run C: minimal online bridge probe
- Run A: small strategy comparison

这些结果是开大 benchmark 之前的最小真实 GPU 闭环，不用于声称 faithful FATE 已完成，也不用于声称 async_release 已真实执行。

## Run B

目录：

- `outputs/online/4gpu_expert_trace_20260710_090102/run_b_expert_trace`

结论：

- `rank0..3_source_expert_counts.jsonl` 全部非空
- 16 个 layer 都有记录
- `expert_trace_collection_passed=true`
- `expert_trace_warnings.jsonl` 当前为空

重建结论：

- `expert_trace_available=true`
- `source_expert_records_count=64`
- `merged_layer_count=16`
- `complete_world_matrix_layer_count=16`
- `incomplete_world_matrix_layer_count=0`
- `mean_relative_l1_error=0.9355505261940585`
- `expert_to_traffic_mapping_valid=true`
- `source_rank_granularity_required=true`
- `recommended_next_predictor_direction=source_rank_expert_prediction`

解释：

- world merge 和 trace schema 已经打通
- 但 O1 平均误差仍高，说明 expert-to-traffic 对齐/标定还没到可直接声称 contribution 2 成立的程度
- 下一步仍应先围绕真实 expert trace 做 reconstruction audit，再决定是否实现真实 gate replay predictor

## Run C

目录：

- `outputs/online/4gpu_bridge_probe_20260710_091120`

结论：

- `birkhoff_phase_local` 通过
- `routersense_joint_priority_phase_sync` 通过
- `execution_audit_status=passed`
- `watchdog_report.status=not_triggered`
- `transport_execution.jsonl` 非空
- policy 名称真实进入 runtime，没有 silent fallback

当前 bridge probe 说明：

- `routersense_joint_priority_phase_sync` 已能在真实 4GPU phase_sync 路径中落到计划与执行 artifact
- 这仍不等于 safe-U 已完成正式 online benchmark

## Run A

目录：

- `outputs/online/4gpu_strategy_compare_20260710_171849`

最小策略集：

- `disabled`
- `birkhoff_phase_local`
- `routersense_joint_priority_phase_sync`

当前 rank0 total forward:

- `disabled`: `5186720.655 us`
- `birkhoff_phase_local`: `5157683.672 us`
- `routersense_joint_priority_phase_sync`: `7405063.436 us`

当前含义：

- 这轮只说明最小 4GPU 路径已经能真实比较这三条线
- `routersense_joint_priority_phase_sync` 当前明显慢于 `birkhoff_phase_local`
- 下一步应优先分析 planning/control timing，而不是直接扩成大 benchmark

## Hard Truths

- `gpu_not_run=false`
- `async_release_real_collectives_not_run=true`
- `faithful_fate_not_validated=true`
- `run_b_debug_trace_not_for_performance=true`
- `large_benchmark_not_run=true`

## Next Step

下一轮优先顺序：

1. 先分析 Run B expert-to-traffic reconstruction
2. 再分析 Run C / Run A 的 planning timing 与 control overhead
3. 修正 bridge 的 prediction/control 成本后，再决定是否扩大策略对比
