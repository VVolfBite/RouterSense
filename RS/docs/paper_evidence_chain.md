# Paper Evidence Chain

这个文档把 RouterSense 论文的三条核心贡献和当前代码入口对齐。

## Claim 1

**Multi-phase joint scheduling has real optimization space**

对应代码：

- `experiments/offline/build_replay_fixture_from_control_trace.py`
- `experiments/offline/run_real_trace_evidence_suite.py`
- `experiments/offline/run_transport_stress_replay.py`
- `src/rs/scheduling/`
  - `B_birkhoff_wave`
  - `U_gated_maxweight_matching`
  - `U_barrier_criticality_global_matching`

当前状态：

- 已能把真实 online trace 转成 replay fixture。
- 已能按 pairing-first 口径比较 B / raw U / safe U：
  - `B_gated_greedy_maximal` vs `U_gated_greedy_maximal` vs `RS_safe_gated_greedy`
  - `B_gated_maxweight_matching` vs `U_gated_maxweight_matching` vs `RS_safe_gated_maxweight`
  - `B_barrier_criticality_matching` vs `U_barrier_criticality_global_matching` vs `RS_safe_barrier_criticality`
- 已能在 offline execution-window 语义下比较 `B_birkhoff_wave` 与 raw/safe `U_*`。
- 已能把真实 fixture 进一步压成 communication-only transport-stress replay 报告。
- 当前 main safe-U 主线是：
  - `RS_safe_barrier_criticality`
  - `RS_safe_gated_greedy`
- 已新增 `routersense_joint_priority_phase_sync` 与 `routersense_joint_async_release_sim` 作为 bridge candidate，用来把 `U_*` 空间往 online 语义推进。
- 这条证据链当前是 offline 可验证，不等于 online 已实现。

## Claim 2

**Cross-layer prediction can improve scheduling and reduce online cost**

对应代码：

- `experiments/offline/run_real_trace_evidence_suite.py`
- `experiments/offline/replay_fixture_policy_study.py`
- `src/rs/runtime/online/megatron_ep/pending_window/`
- `src/rs/runtime/online/megatron_ep/async_release/`

当前状态：

- offline 端已能区分：
  - `zero_hint`
  - `copy_current_dispatch`
  - `fate_style_history`
  - `fate_style_linear`
  - `perfect_trace`
  - `actual_trace`
- 但这些里只有前两类和 traffic-matrix baseline 已经实现；
  faithful FATE-style gate replay predictor 还没有真实 router replay。
- online 侧现在已经有 tensorized dispatch-matrix gather 和 lightweight predictor contract：
  - `ZeroHintPredictor`
  - `CopyCurrentDispatchPredictor`
- offline 侧当前只有 traffic-matrix baseline：
  - `FATEStyleHistoryPredictor`
  - `FATEStyleLinearTrafficPredictor`
- 已新增 expert foundation：
  - `expert_trace.py`
  - `expert_to_traffic.py`
  - `expert_evaluation.py`
  - `gate_replay_predictor.py`
- 已有 `run_prediction_replay_suite.py`，可把预测矩阵真正灌入下一层 replay，并比较：
  - `zero_hint`
  - `copy_current_dispatch`
  - `fate_style_history`
  - `fate_style_linear`
  - `perfect_trace`
  - `actual_trace`
- 已有 `run_expert_to_traffic_reconstruction.py`，用于 GPU expert trace 到位后先回答：
  - O1 actual source-expert -> actual traffic
  - O2 global expert counts -> traffic
  - O3 current source-expert copy -> next traffic
  - O4 current traffic copy -> next traffic
- 当前 `prediction` 证据链可以比较：
  - predictor 误差
  - predicted-P2 下 safe-U makespan
  - oracle P2 上限
- 当前 expert trace 如果不存在，suite 必须明确输出：
  - `expert_trace_available=false`
  - `gpu_collection_required=true`
- 当前 `MockGateReplayPredictor` 仍然只是 mock/contract：
  - `faithful_fate_style=false`
  - 不能进入 paper claim
- `run_prediction_replay_suite.py` 现在必须对
  `zero_hint` / `copy_current_dispatch` / `perfect_trace` / `actual_trace`
  统一输出真实 prediction error，而不是默认 0
- prepared-plan 应消费 `predicted_next_dispatch`，而不是把 gathered current matrix 本身写成 predictor。
- online 端当前只有 phase_sync 下的 prediction-aware policy。
- 但这还不等于真实 online predictor 已完成部署：当前 FATE-style predictor 主要服务 offline artifact、预测误差分析和 future async-release 设计。
- 当前真实 fixture 上，prediction replay 可能仍不能稳定优于 `zero_hint`；这必须如实报告，不能把接口闭环写成已实现收益。

当前最准确的结论应写成：

> Current predictors can approximate rank-to-rank traffic shape, but do not yet validate expert-level FATE prediction.
> The next evidence step is to collect expert route traces, predict source-rank x expert counts, and then reconstruct/calibrate expert-derived traffic matrices.

## Claim 3

**RouterSense is a real reproducible online runtime**

对应代码：

- `src/rs/runtime/online/megatron_ep/`
- `experiments/online/run_strategy_comparison.py`
- `experiments/online/support/runtime_presets.py`
- `experiments/offline/replay_online_control_trace.py`

当前状态：

- `phase_sync` 是当前真实可执行、可审计、可 replay 的 online 主线。
- replay trace、execution audit 和 public runtime surface 已经打通。
- `async_release` 现在已经从纯 dataclass skeleton 推进到：
  - CPU executable simulator
  - `AsyncReleaseExecutionPlan`
  - `AsyncReleasePlanBuilder`
  - compiled tensor schedule
  - tensor-only agreement helper
  - fail-closed executor skeleton（默认关闭）
- 但还没有 GPU executor integration。

因此当前论文口径必须保持诚实：

- online reproducible runtime 已成立；
- full async joint execution 仍是下一阶段工作；
- `async_release simulator` 只能证明语义路径和潜在 hidden-cost 机制，不能替代真实 online executor 结果。
