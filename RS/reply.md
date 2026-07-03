# N19：生成 AR Answer Request 报告

## 任务

基于当前 `report.md` 和 N18 诊断上下文，生成一份结构化的 **AR Answer Request** 报告，供其他模型逐条回答。

输出路径：`RS/ar_answer_request.md`

---

## 报告结构要求

```markdown
# AR Answer Request

## Context
- 项目：RouterSENSE 分布式 MoE 调度系统
- 核心命题：联合调度（U_gated）优于独立调度（Birkhoff）——POC1 离线已验证
- 当前问题：真实 2-node 分布式推理中，调度 throughput 未超越 native baseline
- 环境：2 node × 1 GPU, OLMoE-1B-7B-0924-Instruct, 64 samples, world_size=2

## Q1：开销分解
【背景】birkhoff+wave 的 scheduled comm（2.75ms）< native comm（3.41ms），调度确实减少了通信量，但 throughput 仍输 6%（5.39 vs 5.74 samples/s）。
【请求】量化以下各项在 per-sample 总时间中的占比：
- pack/unpack GPU 开销
- NCCL 启动延迟（每次 all_to_all_single 的固定开销 × wave 数量）
- control_plane 开销（report 显示 7-13ms，需要解释其构成）
- 调度器求解时间（planner_ms）

## Q2：control_plane_ms 构成
【背景】native_baseline 的 control_plane_ms = 6.39ms，scheduled 为 7-13ms。native 不走调度路径却有 6.39ms。
【请求】解释 control_plane_ms 包含哪些操作。列出代码路径和大致耗时分布。

## Q3：2×2 矩阵理论增益上限
【背景】POC1 在 8 GPU（8×8 矩阵）上 joint_gain=7pp。当前 2 GPU 只有 2×2 流量矩阵。
【请求】
- 在 2×2 流量矩阵下，最优调度相比 naive all-to-all 的理论通信量减少上限是多少？
- Birkhoff 在 2×2 下是否已经是最优解？联合调度是否还有额外空间？
- 给出数学推导或反例。

## Q4：波数与 NCCL 调用次数
【背景】wave 粒度下，每 wave 一次 all_to_all_single。
【请求】
- 当前 64-sample/2-GPU/OLMoE-1B 场景下，dispatch + combine 各产生多少个 wave？
- 每个 wave 的平均 NCCL 通信量（bytes）是多少？
- NCCL 启动延迟按 0.5ms 估算，多 wave 的总额外启动开销是多少 ms？

## Q5：代码优化空间
【背景】wave_executor.py 中每个 wave 执行 pack → all_to_all_single → unpack，涉及 GPU gather/scatter 和 .clone()。
【请求】
- 评估 pack/unpack 的可优化空间（避免 clone、in-place scatter、cuda graph 等）
- 估算优化后 per-wave 开销能降低多少 ms
- 优化后 scheduled 能否超越 native？

## Q6：实验设计建议
【背景】需要在有限资源（2 node × 1 GPU）下最大化证明联合调度增益的机会。
【请求】
- 给出最小可行实验配置（batch size × 模型 × layer 数量）使调度增益可观测
- 如果 2 GPU 不够，给出最低 GPU 数阈值
- 是否可以通过增大 hidden_size 或 expert 数量（不换模型）来扩大通信面？

## Q7：POC1 makespan 模型修正
【背景】POC1 的 improvement_pct 基于纯 makespan 模型，不含 NCCL 启动、pack/unpack。
【请求】
- 如何在 makespan 模型中引入 NCCL 启动开销参数（α_ms）和 pack/unpack 开销参数（β_ms/wave）？
- 用真实数据（2.75ms scheduled comm, ~2ms pack/unpack 估算）反推 α 和 β 的值
- 修正后的模型是否仍能得出「联合调度优于独立调度」的结论？
```

---

## 执行指令

读取以下文件后生成 `ar_answer_request.md`：

1. `RS/report.md` — 当前实验结果
2. `RS/src/rs/runtime/distributed_ep/adapter/runner.py` — 调度注入入口
3. `RS/src/rs/runtime/distributed_ep/core/wave_executor.py` — wave 执行细节
4. `RS/src/rs/runtime/distributed_ep/core/wave_planner.py` — wave 规划
5. `archive/backup/20260703_multimodel_u_scheduler_snapshot/artifacts/olmoe_n8_s64_summary.json` — POC1 基准

生成后写入 `RS/ar_answer_request.md`。

## 格式要求

- 每个 Q 独立成节，包含「背景」和「请求」两部分
- 请求必须是具体的、可量化回答的问题（禁止「请分析」类开放问题）
- 所有数值引用标注来源文件
- 报告长度控制在 200 行以内
