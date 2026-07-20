# Predictor Contract

这个文档只定义 RouterSense 论文贡献 2 所需的 predictor 接口边界。

它的目标不是实现真实 predictor，而是明确：

- 什么东西算 predictor 输入
- 什么东西算 predictor 输出
- 离线 `p2_source` 的几种语义分别对应什么
- 什么不应该被误写成 predictor

当前必须额外强调：

- 现有 `fate_style_history` / `fate_style_linear` 是 traffic-matrix baseline
- 它们不是 faithful FATE-style gate replay predictor
- faithful FATE 的下一步证据链必须先建立 expert-route trace 和 expert-level prediction
- expert trace collection 是 debug collection path，必须使用 `observation.profile=debug` 和 `observation.capture_expert_trace=true`

## 目标

predictor 的职责是：

- 在当前 layer 执行期间，尽早给出“下一层 expert activation / dispatch”的预测
- 在 `layer l` 的 dispatch 或 gate 观测点启动预测
- 在 `layer l+1` 的真实 dispatch 到来时，对上一层预测做误差审计
- 让 runtime 可以提前构建 shadow / prepared plan
- 在不显著增加在线控制面成本的前提下，逼近 oracle prediction 的调度收益

predictor 的职责不是：

- 直接执行通信
- 绕过 root-authoritative agreement
- 把 offline oracle upper bound 伪装成在线预测器

## Expert-First Predictor Input

最小输入应当包含：

- `layer_id`
- `current_source_expert_counts`
- `current_selected_experts`
- optional `routing_weights`
- `ep_group_size`
- `model_id_digest`
- `trace_id_digest`

可选输入：

- `routing_summary`
- `previous_layer_history`
- `request_digest`
- `microbatch_id`
- `expert_placement_digest`

这些输入描述的是：

- 当前层真实已观测到的 expert routing
- 以及可用于推断下一层 dispatch 的轻量上下文

traffic-matrix predictor 可以作为 baseline，但它不应取代 expert-first predictor contract。

## Predictor Output

最小输出应当包含：

- `predicted_source_expert_counts`
- `predicted_next_dispatch_matrix`
- `confidence`
- `predictor_name`
- `predictor_version`
- `prediction_digest`
- `oracle`
- `evaluation_eligible`

语义：

- `predicted_next_dispatch_matrix`
  - 下一层 dispatch 的预测矩阵
- `confidence`
  - 预测可信度
- `oracle`
  - 是否来自 oracle / perfect trace
- `evaluation_eligible`
  - 是否可进入正式线上/离线可比较结果

## Offline `p2_source` Mapping

当前离线分析里的 `p2_source` 应按下面理解：

- `zero_hint`
  - 不使用跨层预测
  - 作为 no-prediction baseline

- `copy_current_dispatch`
  - 极便宜 heuristic predictor
  - 用当前 dispatch 近似下一层 dispatch

- `fate_style_history`
  - 基于历史层统计 / EWMA 的 traffic-matrix baseline
  - 不等于 faithful FATE gate replay

- `fate_style_linear`
  - 基于 `[D_{l-1}, D_l, R_l] -> D_{l+1}` 的 traffic-matrix baseline
  - 当前用于 offline artifact、预测误差分析和 predicted-plan replay

- `gate_replay`
  - faithful FATE-style predictor family 的接口名
  - 当前只有 `MockGateReplayPredictor` / contract / CPU skeleton
  - `faithful_fate_style=false`
  - 真实实现需要 GPU 采集 router input 和 next-layer gate

- `perfect_trace`
  - oracle predict upper bound
  - 来自真实下一层 trace
  - 不是在线 predictor

- `actual_trace`
  - execution-window 语义下的真实第三阶段流量
  - 用于 offline joint scheduling upper bound
  - 不是在线 predictor

- future `real_predictor`
  - 后续真正要接入的预测器输出

## 什么不是 Predictor

### `gathered_global_matrix` 不是 predictor

当真实分布式 EP group 可用时，prepared-plan 的 P2 矩阵现在可以来自：

- `gathered_global_matrix`

这表示：

- 已经修正了“只看单 rank 再复制”的全局矩阵来源问题

但它不表示：

- 已经具备真实 next-layer predictor

原因很简单：

- `gathered_global_matrix` 只是把当前已观测到的分布式矩阵正确收集成全局矩阵
- 它没有完成“预测下一层未来流量”这件事

### `perfect_trace` / `actual_trace` 不是在线 predictor

这两者只应该用于：

- offline oracle upper bound
- execution-window upper bound
- prediction upper bound study

不能把它们写成：

- 当前 online predictor 已经实现

## Expert Route Trace Foundation

本轮之后，contribution 2 的正确地基应拆成三层：

1. `expert_trace.py`
   - 采集 / 表示 `selected_experts`, `routing_weights`, `source_rank`
   - 聚合成 `source_rank x expert_id counts`
2. `expert_evaluation.py`
   - 先评估 expert prediction 误差
3. `expert_to_traffic.py`
   - 再把 expert prediction 映射成 traffic matrix
   - 分开报告 mapping/calibration 误差

如果没有真实 expert trace，就不能声称 faithful FATE predictor 已完成。

## 与三条论文贡献的关系

- 贡献 1：
  - 用 offline joint scheduling 证明 multi-phase space 存在

- 贡献 2：
  - 用 `zero_hint` / `copy_current_dispatch` / traffic-matrix baseline / future gate-replay predictor 比较预测价值
- 当前代码入口：
    - `src/rs/runtime/offline/prediction/`
    - `experiments/offline/train_fate_style_predictor.py`
    - `experiments/offline/evaluate_fate_style_predictor.py`
    - `experiments/offline/analyze_prediction_audit.py`
    - `experiments/offline/run_prediction_replay_suite.py`
    - `experiments/offline/run_expert_to_traffic_reconstruction.py`

- 贡献 3：
  - 在真实 online runtime 中，把 predictor 输出接成 prepared / shadow plan 输入
  - 当前 mainline 已有 tensorized dispatch gather 和 lightweight `zero_hint` / `copy_current_dispatch` predictor
  - 但真实学习式或更强 predictor 仍未接入

## 当前 CPU/offline 闭环

本轮之后，贡献 2 的 CPU/offline 主线应当是：

1. `run_expert_to_traffic_reconstruction.py`
   - 检查 expert trace 是否存在
   - 若存在，验证 expert->traffic 映射误差
   - 对比：
     - O1 actual source-expert -> actual traffic
     - O2 global expert counts -> traffic
     - O3 current source-expert copy -> next traffic
     - O4 current traffic copy -> next traffic
2. `train_fate_style_predictor.py`
   - 当前仍只训练 traffic-matrix baseline
3. `evaluate_fate_style_predictor.py`
   - 输出 baseline traffic predictor 误差指标
4. `run_prediction_replay_suite.py`
   - 把 predicted traffic 真正灌入下一层 replay problem
   - 但必须标明 expert trace 是否存在
   - 现在也必须对 `zero_hint` / `copy_current_dispatch` / `perfect_trace` / `actual_trace`
     统一输出真实 prediction error，而不是默认 0
5. `estimate_planning_hiding_window.py`
   - 粗估 prediction / planning / artifact build 是否可能被 layer interval 隐藏

如果这些脚本显示：

- predictor 误差下降；
- predicted replay 比 `zero_hint` 更接近 `perfect_trace`；
- planning cost 相对 layer interval 很小；

才可以说贡献 2 开始接近成立。

当前更准确的口径应是：

> Current predictors can approximate rank-to-rank traffic shape, but do not yet validate expert-level FATE prediction.
> The next evidence step is to collect/replay expert route traces, evaluate source-rank x expert prediction, and then calibrate expert-derived traffic matrices.
