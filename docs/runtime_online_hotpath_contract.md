# RouterSense Online Hot Path Contract

这个文档只约束 `src/rs/runtime/online/megatron_ep/` 的在线热路径，不讨论离线分析。

## 允许进入 perf hot path 的内容

`perf` profile 下，在线运行时默认只保留执行必需和最小观测：

- 构造当前 phase 的本地 `PhaseReadyContext`
- 编码本 rank 的轻量 planning summary
- root 收集全局摘要并生成 wave-level 计划
- 广播 abstract plan
- 每个 rank 本地 materialize 为 `PhaseExecutionPlan`
- executor 按统一 wave 顺序执行
- 最小 timing / audit counters
- 轻量 control replay trace
- tensorized traffic-matrix gather for dispatch/prediction bookkeeping

这些步骤必须保持：

- root-authoritative
- wave-level plan
- 不修改 payload layout / offsets / tensor roles
- 不提前执行未 ready 的 phase
- 不在 P2 / prediction / prepared-plan matrix 路径中使用 Python object collectives

## perf hot path 默认不允许做的事

下面这些内容只允许在 `execution` 或 `debug` profile 中保留，不能默认进入 `perf` 热路径：

- full phase context JSON
- full transport bundle JSON
- full scheduled plan dict
- pending-window shadow
- prepared phase shadow
- tensor capture
- full bucket order / full wave edges
- 大型 debug artifact dump
- plotting / 离线报告生成
- `dist.all_gather_object` / `dist.gather_object` / `dist.broadcast_object_list` 用于 P2 / prediction / prepared-plan matrix

原则是：

- `perf` 只保留控制面规模、计划规模、最小时序和 correctness counters
- `execution` 才保留完整执行观测
- `debug` 才保留最详细诊断

## fast path 的安全边界

当前 fast path 只是“更薄的当前 phase 计划生成”，不是本地 greedy，也不是绕过协商。

它必须继续满足：

- 仍然通过 root 生成全局一致的 wave-level plan
- 仍然广播 abstract plan
- 仍然由各 rank 本地 materialize
- 仍然走现有 executor

它不能做：

- 本地 rank 自己决定最终发送顺序
- 改 executor collective 语义
- 改 payload layout
- 提前执行未 ready 的 P1/P2

## online / offline 边界

在线 runtime 负责：

- 真实观测
- 真正的计划生成与执行
- 最小 timing / audit
- 轻量 replay trace 落盘

offline 负责：

- trace 重建
- 控制面重放分析
- 波次规模 / wire size 统计
- 理论对比和报告

如果某项分析不需要真实 GPU 执行，就不应该继续堆进 online hot path。

## Prediction / P2 Matrix Rule

online runtime 里的 traffic matrix gathering 必须是 tensorized / NCCL-friendly：

- 允许：
  - `all_gather_into_tensor`
  - tensor `all_gather`

- 不允许：
  - Python object collective 来构建 P2 / prediction / prepared-plan matrix

`gathered_global_matrix` 也不能被误写成 predictor。
它只是“当前已观测流量矩阵的全局构造方式”，不是 next-layer prediction。
