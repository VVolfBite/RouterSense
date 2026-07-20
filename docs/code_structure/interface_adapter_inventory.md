# Interface Adapter Inventory

本文件只记录第二阶段接口统一后，各 adapter 的角色，不评价算法正确性。

## 正式必要

- `rs.scheduling.unified_interface.SchedulingPolicy`
- `rs.runtime.online.megatron_ep.compiler_facade.ScheduleCompiler`
- `rs.runtime.online.megatron_ep.execution.executor_facade.TransportExecutor`

## 临时 bridge

- `LegacyLogicalPolicyAdapter`
  作用：把旧 `build_logical_plan(problem)` 桥接到新 `plan(request)`
- `compile_prepared_window_phase_plan`
  作用：把 prepared logical plan 编译到当前 phase physical plan
- `MultiphasePendingWindowAdapter`
  作用：旧 joint pending-window runtime bridge，当前仍被 runtime compiler facade 间接依赖

## diagnostic-only

- `NativeOrderPolicy`
- `NativePassthroughIdentityPolicy`
- `JointShadowP0P1Policy`

这些 policy 只用于 shadow/debug，不属于正式 offline/online canonical policy 构造入口。

## 本阶段未删除

- 旧 `src/rs/scheduling/base.py` Protocol 仍保留
- 旧 pending-window adapter 仍保留
- 删除数量：0

原因：

- 仍有测试、bridge 或 runtime 路径引用
- 本阶段只做“名称与外部接口统一”，不做大规模 runtime 语义切换
