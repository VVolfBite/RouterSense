# Predictor Contract

这个文档只定义 RouterSense 论文贡献 2 所需的 predictor 接口边界。

它的目标不是实现真实 predictor，而是明确：

- 什么东西算 predictor 输入
- 什么东西算 predictor 输出
- 离线 `p2_source` 的几种语义分别对应什么
- 什么不应该被误写成 predictor

## 目标

predictor 的职责是：

- 在当前 layer 执行期间，尽早给出“下一层 dispatch 流量”的预测矩阵
- 让 runtime 可以提前构建 shadow / prepared plan
- 在不显著增加在线控制面成本的前提下，逼近 oracle prediction 的调度收益

predictor 的职责不是：

- 直接执行通信
- 绕过 root-authoritative agreement
- 把 offline oracle upper bound 伪装成在线预测器

## Predictor Input

最小输入应当包含：

- `layer_id`
- `current_p0_dispatch_matrix`
- `current_p1_return_matrix`
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

- 当前层真实已观测到的通信
- 以及可用于推断下一层 dispatch 的轻量上下文

## Predictor Output

最小输出应当包含：

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

## 与三条论文贡献的关系

- 贡献 1：
  - 用 offline joint scheduling 证明 multi-phase space 存在

- 贡献 2：
  - 用 `zero_hint` / `copy_current_dispatch` / `perfect_trace` / future `real_predictor` 比较预测价值

- 贡献 3：
  - 在真实 online runtime 中，把 predictor 输出接成 prepared / shadow plan 输入
  - 但当前 mainline 还没有真实 predictor，只完成了 gathered global matrix 和 prepared-plan 接线
