**Runtime Cleanup Inventory**

当前正式 runtime 主链：
`PhaseReadyContext -> CanonicalBucketTask -> SchedulingPolicy.plan() -> LogicalSchedulePlan -> ScheduleCompiler.compile() -> PhaseExecutionPlan -> TransportExecutor.execute() -> ExecutionResult`

已完成：
- `PreparedWindowRuntimeState` 取代松散 `_prepared_plan_state`。
- runtime compile 现在传入真实 canonical tasks，而不是空元组。
- direct compiler 已接管 prepared-plan 正式执行路径。
- preflight fallback 已通过统一 executor facade。
- legacy scheduling protocols 已迁到 [legacy_interfaces.py](/root/autodl-tmp/RouterSense/RS/src/rs/scheduling/legacy_interfaces.py)。

仍待收尾：
- `pending_window` / `async_release` 旧 shadow 模块命名空间清理。
- `lifecycle.py` 内部 planning/export 逻辑迁出到 service 模块。
