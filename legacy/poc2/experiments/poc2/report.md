# POC2 阶段报告

## 目标

POC2 的目标是把 POC1 的分析器升级成一个真正可执行的单机多卡调度原型，并回答：

- scheduler 是否真的进入真实执行链路
- policy 是否真的改变 NCCL dispatch/return 的关键瓶颈
- dependency-derived information 是否在强 state baseline 之外有独立价值

## 已完成内容

### POC2 第一阶段

- 统一调度器接口
- `fifo / state-only / dependency-only / full`
- 单机 runner
- 真实 routing 接入
- 真实模型支持：
  - `allenai/OLMoE-1B-7B-0924`
  - `Qwen/Qwen1.5-MoE-A2.7B`

### POC2 第二阶段

- synthetic service backend 体系
- `mock-sleep`
- `synthetic-matmul`
- `synthetic-token-linear`
- transfer-aware backend
- artifact 导出与结果归档

### POC2 第三阶段以后

- 真实 4 GPU NCCL harness
- GPU↔GPU `all_to_all_single` dispatch/return
- rank-local expert-like MLP compute
- paired benchmark protocol
- strategy order / position effect 统计
- strong-state / Lina-inspired / shuffled controls
- information gain diagnosis
- runtime leverage audit

主要入口：

- [nccl_4gpu_benchmark.py](/root/autodl-tmp/RouterSense/experiment/poc2/nccl_4gpu_benchmark.py)
- [information_gain_diagnosis.py](/root/autodl-tmp/RouterSense/experiment/poc2/information_gain_diagnosis.py)
- [audit_policy_leverage.py](/root/autodl-tmp/RouterSense/experiment/poc2/audit_policy_leverage.py)

## 已得到的关键结论

### 关于 dependency 本身

- dependency score 与 token count / payload bytes / state / strong-state 高度相关
- top-k bucket overlap 基本达到 `1.0`
- `dependency-only` 不能稳定击败 `shuffled-dependency`
- `full` 能稳定赢 `FIFO`，但通常不能稳定赢 `strong-state` 或 `Lina-inspired`

当前判断：

> 在当前 execution model 下，dependency signal 还不能被稳定分离为强 observable state 之外的独立收益来源。

### 关于 runtime leverage

- scheduler 的 release order 确实进入了执行链路
- planned split sizes 与 actual NCCL split sizes 一致，没有发现 layout 忠实性 bug
- 但 `full / random-order / strong-state` 对 round-level bottleneck rank、split-size profile、max-rank bytes 的影响非常接近
- 多数情况下，policy 改变了 bucket identity，但没有稳定改变真正决定 E2E 的 rank-level bottleneck layout

当前 leverage audit 自动结论是：

> Case A: policy has insufficient leverage under the current round formation and placement.

## 当前边界

POC2 不是 production runtime。

它目前能说明：

- 真实 routing 已进入真实多卡 NCCL harness
- policy 确实参与 release / dispatch / return
- 但当前 round formation + collective barrier 结构给 scheduler 的 leverage 很弱

## 对下一步的建议

不建议继续在当前 POC2 结构上做更大规模 benchmark 或继续微调 dependency 权重。

更合理的下一步是：

- 转向新的 execution structure
- 或转向更明确面向 future demand 的 POC3

如果继续，只应继续研究：

- 更细粒度 round formation
- 更强的 asynchronous overlap
- 能真正改变 future bottleneck 的 execution path
