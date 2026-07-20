# Interface Migration Report

接口检查点：

- checkpoint commit: `454f987010b99cd73e0ec9dd9db42aacf1b1311d`

本阶段完成内容：

1. 新增统一调度接口 `SchedulingPolicy.plan(request)`
2. 新增统一编译接口 `ScheduleCompiler.compile(request)`
3. 新增统一 transport 接口 `TransportExecutor.execute(request)`
4. offline replay 改为通过 `SchedulingRequest -> build_policy(...).plan(...)`
5. runtime joint planning 改为通过统一 policy facade 生成 logical plan
6. runtime transport adapter 改为通过统一 executor facade 调用 phase-sync / async-release
7. runtime async local materialization 改为通过 compiler facade 进入旧 prepared-plan 编译器

离线等价结果：

- 9 个 canonical policy：新旧 wave 顺序与 makespan 一致
- 2 个 reference-only canonical policy 被显式拒绝：
  - `barrier_criticality_posthoc_best`
  - `oracle_local_cp_sat`

物理计划 shadow compare：

- phase-local: 一致
- joint bridge: 一致
- 差异性质：无语义差异；当前 facade 仍调用旧 prepared-plan compiler

runtime 当前状态：

- 已切到统一 compiler facade
- `legacy_secondary_policy_invocation_count = 1`
- 因此仍属于 bridge cutover，不是最终纯 compiler 实现

Gloo 证据：

- 当前环境下重新跑 runtime-integrated Gloo gate 时，两个 worker 在 import/init 后被系统 `SIGKILL`
- 这次失败没有 Python traceback，也没有 preflight/fallback 记录
- 已从上一阶段 handoff tar 恢复稳定通过 artifact：
  - `actual_p0_matrix_unit = rows`
  - `p1_is_exact_transpose = true`
  - `batch_isend_irecv_call_count > 0`
  - `phase_sync_fallback_count = 0`
  - `stored_p1_plan_digest == consumed_p1_plan_digest`

本阶段结论：

- `UNIFIED_INTERFACE_COMPLETE`
- `RUNTIME_COMPILER_CUTOVER_PENDING`

下一阶段只建议做：

1. 去掉 runtime compiler 下层的隐藏二次 phase policy 调用
2. 将 `_prepared_plan_state` 从 dict 收敛到 typed dataclass
3. 在不改算法逻辑的前提下整理 `lifecycle.py` 内部函数结构
