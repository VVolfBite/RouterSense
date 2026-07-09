# Async Release Runtime Contract

这个文档描述 RouterSense `async_release` 主线当前能表达什么、不能表达什么。

## 当前状态

- `runtime.line=async_release` 已经公开声明。
- 真实 online launch 仍然 **not implemented**。
- 当前可用的是：
  - `src/rs/runtime/online/megatron_ep/async_release/`
  - shadow-only contract / state machine
  - validation
  - CPU executable simulator

## 当前可表达的语义

- P0 dispatch 在当前层观测后可以进入 ready/release 候选。
- P1 return 必须等待对应 rank 的 P0 inbound completion + compute delay。
- P2 只作为 forecast / shadow plan 输入，不直接执行当前层真实通信。
- 可以表达：
  - shadow plan ready
  - early release
  - fallback to phase_sync
  - prediction lead time
  - hidden planning fraction

## 当前不能表达的语义

- 不能直接 launch GPU executor。
- 不能替代 `phase_sync` public runtime mainline。
- 不能把 CPU simulator 结果写成真实 online latency。

## 代码入口

- Contract / state:
  - `src/rs/runtime/online/megatron_ep/async_release/contracts.py`
  - `state.py`
  - `validation.py`
- Shadow-only controller:
  - `shadow_controller.py`
- CPU simulation:
  - `simulator.py`
  - `experiments/offline/run_async_release_simulation.py`

## 与论文三条贡献的关系

- Claim 1：
  - async-release 是把 offline joint scheduling 空间转成 online 语义的关键桥梁。
- Claim 2：
  - predictor 目标之一是让 shadow/prepared plan 提前 ready，把 planning/control 成本隐藏在前一层窗口里。
- Claim 3：
  - 当前 phase_sync 已是真实可执行 runtime；
  - async_release simulator 是下一步 executor integration 前的必要中间层，不应夸大为已完成 GPU runtime。
