**Runtime Transport Facade**

正式执行入口是 [executor_facade.py](/root/autodl-tmp/RouterSense/RS/src/rs/runtime/online/megatron_ep/execution/executor_facade.py) 的 `execute_transport()`

后端：
- `phase_sync`
- `async_release`

本轮收敛结果：
- 正常 async 路径走统一 facade。
- 正常 phase-sync 路径走统一 facade。
- preflight fallback 也已改为通过统一 facade，不再由 adapter 直接调用底层 sync executor。
