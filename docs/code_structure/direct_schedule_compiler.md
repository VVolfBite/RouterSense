**Direct Schedule Compiler**

当前 direct compiler 位于 [compiler_facade.py](/root/autodl-tmp/RouterSense/RS/src/rs/runtime/online/megatron_ep/compiler_facade.py)。

当前行为：
- runtime 会基于当前 phase 的真实 row matrix 构造 canonical tasks。
- compiler 先做 legacy bridge 与 direct shadow compare。
- 当 shadow plan 满足：
  `missing_task_count=0`
  `extra_task_count=0`
  `execution_order_matches_legacy=true`
  时，正式执行切到 direct compiled physical plan。

当前 Gloo gate 证据：
- `canonical_task_count > 0`
- `legacy_secondary_policy_invocation_count = 0`
- `compiler_shadow_status = ok`
