## M0 closure summary

1. lifecycle dict/object 崩溃如何修复
   - `lifecycle.py` 的 `_predict_dispatch_matrix()` 不再返回裸 dict。
   - 现在返回不可变 `RuntimePredictionCompatResult`，下游统一用属性访问；写入 runtime state 时显式 `to_dict()`。

2. Runtime 正式使用哪种 prediction result
   - formal predictor 仍返回 `PredictionResult`。
   - lifecycle 过渡层消费 `PredictionResult` 后，转换成明确的 typed compatibility object：`RuntimePredictionCompatResult`。

3. target planner 在三种模式下实际调用次数
   - `LOCAL`: Local=1, Joint=0
   - `JOINT`: Local=0, Joint=1
   - `COMPARE`: Local=1, Joint=1
   - 证据：`tests/contract/megatron_ep/test_target_planner_service.py`

4. COMPARE 是否只执行一次 Local 和一次 Joint
   - 是。`PlannerSelector.select_prebuilt(...)` 已加入，service compare 路径只比较预构建计划，不再二次 `plan()`

5. truth 原先从哪里泄漏
   - `runtime/offline/replay_unified.py` 中 `build_multiphase_problem(...)` 原先把 `execution_truth.p2_truth_rows` 写回 `p2_next_dispatch_forecast_matrix`
   - 这会把真实 P2 注入 formal planning-visible forecast 字段

6. planning_task_digest 和 execution_truth_digest 如何拆分
   - 新增 `bucketize_planning_request(request)`：只读取 `P0 actual + P1 actual + P2 hint`
   - 新增 `execution_truth_digest(execution_truth)`：单独哈希真实执行真值
   - `input_task_digest` 仍保留，但只作为 deprecated compatibility alias

7. truth isolation 测试是否真正构造了不同 truth
   - 是。新 `tests/contract/test_truth_hint_isolation.py` 显式构造不同 `ReplayWindow.p2_truth_rows`
   - 同时验证 `planning_task_digest` 不变、`execution_truth_digest` 改变

8. 正式 estimator 是否完全不依赖 legacy_makespan 决策
   - 是。`CommonCorePlanEstimator` 现在只依据 `WindowPlan + PlanningRequest + PlanningCostModel` 统一计算
   - `legacy_makespan` 不再决定 selector 结果

9. formal 和 legacy makespan 差异
   - formal `estimated_makespan` 使用统一 launch/row-transfer/port/full-duplex/expert-delay 口径
   - legacy 数值仍可保留在 metadata 里做 parity audit，但不会覆盖 formal 结果
   - 本轮没有新增自动回退到 legacy 值的逻辑

10. linear predictor 的训练目标
   - 已修正为 `target_next_dispatch_rows`
   - formal linear predictor 新增 `fit(samples)`，`predict(context)` 不再临时训练
   - 它仍明确标记 `offline_only`

11. expert-route 是真实迁移还是降级为 test-only
   - 本轮选择诚实降级
   - `mock_gate_replay` 保留为 `test_only=true`、`deployable=false`
   - M0 closure 不再声称“deployable expert-route predictor 已迁移完成”

12. RouteToTrafficMapper 如何表达 source rank
   - 保留单源 `map(route_prediction, source_rank=...)`
   - 新增 `map_ranked(RankedExpertRoutes, ...)`，用于显式表达 source-rank x route

13. diagonal traffic 如何处理
   - 正式 mapper 现在默认跳过 diagonal/self traffic，只统计远端通信矩阵

14. exact 和 reference 如何区分
   - `reference_only` 不再等价于 `exact`
   - `execution_model == exact_reference` 或 oracle family 才标记 `exact=True`
   - family 现在正确区分：
     - `reference_local`
     - `reference_joint`
     - `exact_local`
     - `exact_joint`

15. semantic digest 排除了哪些身份和 metadata
   - `PlanningRequest.semantic_digest()` 排除了 `request_id/run_id/forward_id/window_id/source_layer_id/target_layer_id`
   - `WindowPlan.semantic_digest()` 排除了全部 metadata
   - 另新增：
     - `PlanningRequest.identity_digest()`
     - `WindowPlan.audit_digest()`

16. Runtime 是否仍直接 import 旧 unified interface
   - 否。架构测试已覆盖 `rs.scheduling.unified_interface`
   - runtime 当前直接 import 已收敛到 formal `rs.planning` / `rs.prediction` / `rs.core.contracts`

17. 哪些 shim 仍保留
   - `src/rs/planning/runtime_compat.py`
   - 现在只剩 algorithm resolution / phase-policy helper lookup，不再承载 legacy planner construction

18. M0 是否真正满足合并条件
   - 在本轮要求范围内，满足
   - 关键条件已验证：typed lifecycle adapter、compare 单次规划、truth/hint 隔离、formal estimator、linear next-target、expert-route 诚实标注、owner GPU accuracy、exact/reference 分类、semantic digest 分离、runtime direct import gate、targeted tests、offline smoke

## Uncertain-but-executed changes

- `LinearTrafficPredictor` 没有完整迁回旧 offline artifact API；本轮只把“训练目标错误”和“predict 内临时 fit”两件硬缺口修正，并保持 `offline_only`
- `RouteToTrafficMapper` 的 ranked-source 语义已落地，但真实 deployable expert-route predictor 迁移没有在本轮完成，因此只保留 test-only 路径
- `CommonCorePlanEstimator` 的统一模型仍是最小成本模型，不是高保真网络模拟器；这属于本轮允许范围内的统一口径，而不是新 simulator 设计
