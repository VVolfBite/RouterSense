## M0 summary

1. 原有 Predictor 接口数量
   - 至少 3 套可见入口：在线 `PredictionInput/PredictionResultBundle` 路径、离线 rolling predictor 路径、expert-route replay predictor 路径。
2. 原有 Planner/registry/catalog 数量
   - 至少 5 套公开/半公开入口并存：`registry`、`catalog`、`public_catalog`、`algorithm_catalog`、`unified_interface`。
3. 新权威接口位置
   - Prediction: `src/rs/core/contracts/prediction.py` + `src/rs/prediction/*`
   - Planning: `src/rs/core/contracts/planning.py` + `src/rs/planning/*`
4. ExpertRoute 与 TrafficMatrix predictor 区分
   - `PredictionContext` 用联合类型区分。
   - traffic predictor 输出标准化 `PredictionHint`。
   - expert-route predictor 额外输出 `ExpertRoutePrediction`，再通过 `RouteToTrafficMapper` 生成标准化 hint。
5. RouteToTrafficMapper 的唯一职责
   - 仅负责 `expert route -> GPU traffic matrix rows`，不承担 predictor、evaluator、artifact、truth 访问职责。
6. PlannerSelector 三种模式
   - `LOCAL`: 只跑 local planner。
   - `JOINT`: 只跑 joint planner。
   - `COMPARE`: 同时跑 local/joint，再由统一 `CommonCorePlanEstimator` + `PlanningCostModel` 比较。
7. LOCAL/JOINT 是否避免双规划
   - 是。测试已覆盖，两个模式都不再无条件双规划。
8. COMPARE 的统一估算口径
   - 一个 estimator，一个 cost model。
   - 优先读 formal `WindowPlan.metadata["legacy_makespan"]` 以降低 parity 漂移；缺失时再落到统一波次估算。这是本轮明确保留的兼容实现。
9. B/U 历史名称兼容
   - 仅保留在 `src/rs/planning/legacy_aliases.py` 与旧 catalog alias 解释层。
   - 新正式 family 只输出 `baseline/local/joint/exact_local/exact_joint`。
10. truth/hint 泄漏是否完全消除
   - 对 formal planner 入口已消除：`PlanningRequest` 只携带 P0/P1 actual 和 `PredictionHint`。
   - truth 仍存在于 offline replay truth/evaluator 合同中，但不再进入 deployable planner formal contract。
11. offline/runtime 迁移调用者
   - online predictor factory
   - online target planning service
   - lifecycle 中两处在线 P2 hint 构造
   - offline replay engine
   - offline prediction evaluation rolling predictor path
   - runtime algorithm resolution imports
12. 已删除旧文件
   - 本轮没有新增删除。
   - 工作区里已有用户预先删除项，未纳入本轮提交。
13. 保留 shim 及删除时机
   - `src/rs/planning/runtime_compat.py`: 仅重导出旧 phase-policy helper，待 Runtime 下一轮不再依赖 helper re-export 时删除。
   - `rs.scheduling` 作为内部实现层保留，待算法实现继续内迁或完全适配 formal planner 后再收缩。
14. parity 结果
   - 目标 CPU fixture 上 formal prediction/planning parity 通过。
   - offline smoke 成功跑通，`audit_invalid_count = 0`。
15. 下一轮 Runtime 必须处理的问题
   - 生命周期中仍有 legacy-shaped metadata adapter
   - target planning service 仍需把 formal plan 回转成 legacy logical plan
   - `runtime_compat` 仍存在
   - 重复预测、后台队列、collective 顺序、TargetPlanStore、Lifecycle 主流程本轮均未触碰

## Uncertain-but-executed changes

- `LinearTrafficPredictor` 没有继续 import `rs.runtime.offline.*`，而是在 `rs.prediction` 下重写了最小 ridge-linear 逻辑，以满足依赖方向约束。这改变了实现归属，但目标语义保持为 offline-only predictor。
- `CommonCorePlanEstimator` 为压低 parity 风险，优先复用 legacy planner 产出的 makespan metadata；它已经统一到同一 estimator/cost-model 调用口径，但底层数值仍部分借用旧实现输出。
- `lifecycle.py` 本轮没有改主流程，只加入 formal predictor 结果到 legacy dict 的适配层，保留了下游旧字段形状。
- 为了跑现有 offline smoke，补了 `src/rs/experiments/output_schema.py` 对 `traffic.bucket_rows` 列表形状的兼容校验；这是入口校验修复，不涉及调度算法语义。
