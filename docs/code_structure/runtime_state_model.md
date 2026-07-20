**Runtime State Model**

正式 runtime 状态现在由 [runtime_state.py](/root/autodl-tmp/RouterSense/RS/src/rs/runtime/online/megatron_ep/state/runtime_state.py) 定义。

核心对象：
- `PreparedWindowRuntimeState`
- `RuntimeExecutionMetrics`

关键变化：
- `lifecycle/host/gate` 中对 `_runtime_state` 的 `.get()/[]/update()/pop()` 直接访问已收敛为 typed `read/write/remove/merge`。
- artifact 字段继续兼容旧 schema，但业务代码不再直接把 state 当 dict 使用。
