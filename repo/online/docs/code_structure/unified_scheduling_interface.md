# Unified Scheduling Interface

本阶段只统一调度算法的外部接口，不改算法内部评分、排序或 makespan 逻辑。

正式接口位于 [src/rs/scheduling/unified_interface.py](/root/autodl-tmp/RouterSense/RS/src/rs/scheduling/unified_interface.py)：

```python
class SchedulingPolicy(Protocol):
    policy_id: str

    def plan(self, request: SchedulingRequest) -> LogicalSchedulePlan:
        ...
```

`SchedulingRequest` 的约束：

- 只包含 `p0_truth_rows`、`p1_truth_rows`、`p2_hint_rows`
- 不包含 `p2_truth_rows`
- `tasks` 必须来自统一 `CanonicalBucketTask`
- `bucket_rows` 不再直接传给 policy，而是先体现在 `tasks`
- `scheduling_mode`、`information_mode`、`max_waves` 从旧 `MultiPhaseSchedulingProblem.options` 保留到新请求，避免行为漂移

当前状态：

- 9 个 canonical policy 已通过 `build_policy(...).plan(...)` 接入统一接口
- 这 9 个 policy 目前都通过 `LegacyLogicalPolicyAdapter` 桥接到旧 `build_logical_plan()` 实现
- `barrier_criticality_posthoc_best` 与 `oracle_local_cp_sat` 被显式拒绝

这两个 policy 被拒绝的原因：

- `barrier_criticality_posthoc_best` 依赖事后真值，不是 planning-time policy
- `oracle_local_cp_sat` 当前只是 reference/reporting 名称，不是 registry 中可直接构造的统一 policy builder

本阶段保证：

- 新旧名称在已支持 policy 上保持 wave 顺序和 makespan 一致
- 未知 policy 名不会静默降级
- offline 与 online 后续可以共享同一个 `SchedulingRequest` 入口
