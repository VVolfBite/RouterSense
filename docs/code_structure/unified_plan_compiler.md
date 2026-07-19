# Unified Plan Compiler

正式 compiler facade 位于 [src/rs/runtime/online/megatron_ep/compiler_facade.py](/root/autodl-tmp/RouterSense/RS/src/rs/runtime/online/megatron_ep/compiler_facade.py)。

目标接口：

```python
class ScheduleCompiler(Protocol):
    def compile(self, request: PlanCompilationRequest) -> CompilationResult:
        ...
```

输入：

- `logical_plan`
- `local_context`
- `global_contexts`
- `canonical_tasks`
- `phase`
- `tensor_role`
- `rank_context`
- `compilation_options`

输出：

- `execution_plan: PhaseExecutionPlan`
- `audit: CompilationAudit`

本阶段实际状态：

- runtime 已改为经过 `compile_schedule(...)` facade
- facade 目前仍在 bridge 模式下调用 `compile_prepared_window_phase_plan(...)`
- 因此 `legacy_secondary_policy_invocation_count` 仍为 `1`
- 这意味着“统一 compiler 外部接口”已建立，但“runtime 完全 cut over 到纯逻辑计划编译”仍待下一阶段

已验证不变量：

- phase-local bridge 产出的 physical waves 与旧编译路径一致
- joint prepared-plan bridge 产出的 physical waves 与旧编译路径一致
- compiler facade 会把 `compiler_id`、`logical_plan_digest`、`compiled_plan_digest` 注入运行摘要

结论：

- `UNIFIED_INTERFACE_COMPLETE`
- `RUNTIME_COMPILER_CUTOVER_PENDING`
