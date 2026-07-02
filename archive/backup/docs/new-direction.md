# INFOCOM 论文方向规划

## 总体架构：两条线，四个点

```
线1: D10 (流量整形) + D11 (跨层预测)
     → "提前知道流量长什么样 → 更聪明地错峰"

线2: D17 (重要性信号) + D20 (优先级队列)
     → "知道哪些 dispatch 更重要 → 重要的先送"

最终: 线1 + 线2 = 完整方案
     → "既错峰，又优先送重要的"
```

**放弃有损路线**：D15(量化)已被TensorRT-LLM生产实现，有损路线需要NLL验证成本高，全线转向无损。

**核心数据支撑**：POC1 ablation 1024条已验证——expert级skew 24x, 4-GPU median=3.0x, 98%推理步有GPU完全空闲。

---

## 线1: 流量感知调度

### D10: MoE Dispatch Traffic Shaping (流量整形) — 主线

**核心想法**: 不让多个请求的dispatch同时涌入all-to-all，通过控制发送时机平滑流量峰值。

**类比**: 快递窗口排队太长 → 让不同部门分批发，别同时挤上去。不需要预测明天寄多少，只需看当前排队长度决定让谁等一下。

**竞品状态**: ✅ 安全

| 竞品 | 优化层 | 我们的差异 |
|---|---|---|
| FAST (NSDI'26) | transport层: 给定流量矩阵后优化搬运路径 | 应用层: 控制发送时机 |
| Aurora (IEEE'24) | 微观: 单次all-to-all内token传输顺序 | 宏观: 跨请求的dispatch流量平滑 |
| Lina (ATC'23) | 通信调度: 谁先传谁后传 | 流量整形: 主动延迟部分dispatch平滑峰值 |
| NCCL EP (NVIDIA'26) | transport层: 统一dispatch/combine API | 调度层: 何时发、发多少 |
| s-MoE (ICML'25) | 减少通信量: 把token搬到对的GPU | 优化不可避免的通信: 让通信更顺畅 |

**NCCL底层现状**: NCCL用固定调度，对动态skewed workload完全不感知(FAST论文原话)。NCCL EP也只是改善传输效率，不做调度决策。**我们的D10在NCCL之上，完全正交。**

**技术实现**:
- **不需要预测流量** — reactive即可: 观察当前积压量 → 决定pacing
- Burst Pacing: 连续请求触发all-to-all时，按固定间隔分批发送
- Load-Aware Batching: 看各GPU积压量，忙的GPU延迟接收
- POC2的RuntimeState已有`destination_pending_work`, `flow_pending_bytes`

**POC资产利用**: ✅ 高
- POC1 ablation 1024条 → 提取流量矩阵
- POC2 scheduler.py → 新增`TrafficShapingScheduler` ~140行
- POC2 NCCL仿真 → 模拟dispatch时序

**风险**:
- ⚠️ 改进幅度: NCCL底层拥塞控制+我们的pacing叠加效果待验证
- ⚠️ 场景依赖: 跨机Ethernet瓶颈更明显，效果应更好(我们选跨机场景)

**代码量**: ~140行 (TrafficShapingScheduler子类)

---

### D11: Layer-Aware Dispatch Scheduling (跨层预测调度) — 辅助/联合主线

**核心想法**: 用第N层的gate结果预测第N+1、N+2、N+3层的dispatch矩阵，提前规划多层的发送策略。

**类比**: 知道接下来3轮会议分别要叫哪些顾问 → 提前安排走廊和会议室，不用每轮临时找。

**竞品状态**: ⚠️ 需注意差异化

| 竞品 | 用途 | 我们的差异 |
|---|---|---|
| Fate (中山大学+华为, 2025.02) | 单GPU offloading, 97.15%预测准确率 | 分布式EP网络调度 |
| s-MoE (ICML'25) | 减少通信量(89%路由预测), 单机 | 多层联合调度优化 |
| Pre-gated MoE (ISCA'24) | 预测下一层expert, 需改模型架构 | 不改模型, 纯系统层 |
| "Patterns behind Chaos" (ISCA'26) | 证明跨层相关性很强 | 网络调度层面的利用 |

**关键发现**: Fate的97%准确率说明跨层预测可行。预测多层后准确率衰减(97%^3≈91%)，但可能仍够用。

**跨层预测的真实价值定位**:

| 用途 | 收益 | 是否可行 |
|---|---|---|
| 多层联合release plan (一次规划3-4层, 避免greedy局部最优) | **核心** | ✅ 需oracle实验验证 |
| Buffer预分配 (提前分配receive buffer, 消除动态分配开销) | 小 | ✅ 可行 |
| Dispatch plan预计算 (提前算好token排序/打包顺序) | 小-中 | ✅ 可行 |
| ~~Combine-Dispatch pipeline overlap~~ | ~~大~~ | ❌ **数据依赖硬约束, 不可行** |
| 热点expert预取 | 需KV cache | ⚠️ 与Fate重叠 |
| Skip空GPU (98%步有空GPU) | 小 | ✅ 可行 |

**关键纠偏**: 跨层预测**无法**实现Layer N combine与Layer N+1 dispatch的pipeline重叠——因为Layer N+1 dispatch需要Layer N combine的输出(hidden_states)作为输入, 数据依赖是物理约束, 不是工程实现问题。即使有完美预测, 也无法发送不存在的数据。

跨层预测的真正核心价值是**多层联合调度的决策质量提升**: 从per-layer greedy(每层独立最优)到multi-layer global optimal(看到未来3层流量后做全局最优)。典型场景: Layer i和Layer i+2都打到GPU_0, greedy可能让GPU_0在Layer i被打满, 到i+2时还在消化上一批; 联合规划可在Layer i就有意识整节奏。

**POC资产利用**: ⚠️ 中
- OLMoE-1B-7B-0924-Instruct 实测为 16 个返回 router logits 的 MoE/router-active layers，不是先前假设的 3-4 层
- 因此跨层预测与联合调度的状态空间、ILP 复杂度和门限都必须按 16 层重估
- POC1 ablation有4层MoE数据可补充
- POC2 GlobalDispatchPlan可扩展为multi-layer plan
- 需额外验证OLMoE跨层相关性

**风险**:
- ⚠️ 预测准确率多层衰减
- ⚠️ 即使预测准了，网络调度增益可能有限
- ⚠️ OLMoE跨层相关性未验证(Fate在其他模型上验证)
- ⚠️ 降级方案: 如果相关性不够强，放弃D11全力做D10+D20

**验证方法**: 跑一次forward pass，保存相邻层 gate 输入 hidden states；用 `hidden_i -> gate_{i+1}` 做跨层预测，比较 prefetch accuracy 和 gate-input cosine similarity。

**代码量**: ~150行 (CrossLayerScheduler + 跨层分析脚本)

---

## 线2: 优先级感知调度

### D17: Quality-Aware Dispatch Prioritization (重要性信号) — 联合主线

**核心想法**: 找到一个**便宜的**信号，能区分"这个dispatch重要"和"那个不太重要"。

**类比**: 快递分拣时怎么区分"合同"和"广告"？需要一个标签。

**竞品状态**: ✅✅ 最安全
- 无人将expert importance映射到dispatch网络优先级
- AdapMoE: 用sensitivity做expert跳过，不做网络调度
- QLLM (EuroMLSys'25): 请求级优先级，不做dispatch级

**文献调研 — expert importance信号**:

| 信号 | 成本 | 可信度 | 来源 |
|---|---|---|---|
| **TopK rank** | 零 | ✅ | routing softmax排序 |
| **Layer depth** | 零 | ✅ | 深层delay影响更大 |
| **Expert load** | 零 | ⚠️ | 均衡化而非重要性 |
| Expert output L2 norm | 零 | ⚠️ 未验证 | 输出大=贡献大? |
| Token position | 零 | ⚠️ | prefill vs decode差异 |
| Routing weight | 零 | ❌ | INFORM(TMLR'26)证实不靠谱 |
| 二阶灵敏度 (AdapMoE) | 离线一次 | ✅ | loss对expert output的二阶导 |
| Gradient sensitivity (INFORM) | 离线一次 | ✅ | 梯度分析 |
| 压缩误差敏感性 | 离线一次 | ⚠️ | ACM'25 |

**关键文献**:
- **INFORM (TMLR'26)**: "routing dominance is a poor proxy for functional necessity" — 被叫得最多的顾问不一定最重要
- **AdapMoE (ICCAD'24)**: 二阶灵敏度区分expert重要性
- **Compression Error Sensitivity (ACM'25)**: 不同expert压缩误差对质量的影响

**降级方案**: 不做"质量感知"，改做"**紧急度感知**"
- 信号: topK rank + layer depth + expert load
- 不依赖routing weight，基于调度逻辑
- 论文改称"urgency-aware scheduling"

**POC资产利用**: ⚠️ 中
- POC1有routing weight + delta_nll → 可挖掘更多信号
- POC2 BucketRecord有`estimated_dispatch_cost`, `is_hot_bucket`
- 核心问题: 哪个信号真正有效需要对比实验

**风险**: ⚠️ 中高
- 最可靠的信号(gradient sensitivity)太贵
- 便宜的信号(topK rank)效果可能有限
- 如果所有便宜信号都不work，D20的收益也很小

**代码量**: ~100行 (信号提取 + importance scoring模块)

---

### D20: Priority Queue for All-to-All (优先级队列) — 联合主线

**核心想法**: 在all-to-all dispatch中引入优先级队列——高优先的先发，低优先的后发或合并。

**类比**: VIP通道——重要文件先送，普通文件凑一批。

**竞品状态**: ✅ 安全
- FAST: 调度时机，不管优先级
- Aurora: 单次all-to-all内传输顺序(微观)
- 无人做dispatch级优先级队列

**与D17的关系**: D20是D17的**下游** — D17提供信号，D20提供机制

**技术实现**:
- POC2 scheduler.py继承BaseScheduler，override `score_bucket()` 即可
- 即使信号简单(topK rank)，机制本身有价值

**可用优先级信号(零成本)**:

```
1. rank_in_topK        → 第几选择（0=最爱, 1=次选）
2. layer_index L       → 第几层（深→更紧急）
3. expert_load[j]      → expert_j当前积压量（少→先送，均衡化）
4. token_position      → token在序列中的位置
5. is_first_dispatch   → 是不是该token的第一次dispatch

组合信号:
6. rank × (1/layer)    → 第一选择 + 深层 = 最高优先
7. expert_load倒数     → 越闲的GPU越先送
```

**POC资产利用**: ✅ 高
- POC2 scheduler.py → 新增`PriorityScheduler` ~50行
- 直接继承已有多策略框架

**风险**: ⚠️ 依赖D17信号质量，但机制本身无风险

**代码量**: ~50行 (PriorityScheduler子类)

---

## 组合实验设计

| 实验 | 内容 | 验证目标 |
|---|---|---|
| Exp 0 | FIFO baseline | 基准线 |
| Exp 1 | D10 alone (traffic shaping) | 错峰是否有效 |
| Exp 2 | D10 + D11 (加跨层预测) | 预测是否增强shaping |
| Exp 3 | D10 + D20 (加优先级) | 优先级是否有效 |
| Exp 4 | D10 + D11 + D20 (全组合) | 完整方案效果 |
| Exp 5 | vs FAST baseline | 与transport层方案对比 |
| Exp 6 | vs Aurora baseline | 与微观方案对比 |

**对比维度**: E2E latency, tail latency (P99), GPU utilization fairness, throughput

---

## 时间线: 6/29 → 7/31 (4.5周)

| 周 | 任务 | 交付 | 关键检查点 |
|---|---|---|---|
| **W1** (7/1-7/7) | 流量矩阵提取 + 跨层相关性验证 + D10编码 | 数据ready + D10代码 | ✅/❌ 跨层相关性是否够强 |
| **W2** (7/8-7/14) | D10实验 + 优先级信号探索 + D20编码 | D10结果 + D20代码 | ✅/❌ D10改进幅度是否够大 |
| **W3** (7/15-7/21) | D11跨层调度 + D17信号确定 + 组合实验 | 全部实验数据 | ✅/❌ 信号是否有效 |
| **W4** (7/22-7/28) | 论文写作 + 补充实验 | 论文初稿 | — |
| **W4.5** (7/29-7/31) | 润色 + 最终检查 | 提交 | — |

**降级路径**:
- W1跨层相关性弱 → 放弃D11，W3全力优化D10+D20
- W2 D10幅度小 → 调整pacing参数，或聚焦跨机场景增强效果
- W3信号太弱 → D17降级为urgency-aware，放弃quality-aware叙述

---

## 代码工作量汇总

| 模块 | 行数 | 复杂度 | 依赖 |
|---|---|---|---|
| 流量矩阵提取 | ~100行 | 低 | POC1数据 |
| 跨层相关性分析 | ~150行 | 低 | POC1数据 |
| TrafficShapingScheduler | ~140行 | 中 | POC2框架 |
| CrossLayerScheduler | ~150行 | 中 | POC2框架 |
| PriorityScheduler | ~80行 | 低 | POC2框架 |
| 信号提取模块 | ~100行 | 中 | POC1数据 |
| **合计** | **~720行** | — | — |

---

## 立题评审

### 审稿提示词

你是一位计算机网络与系统方向的资深审稿人，目标会议为 IEEE INFOCOM。请对以下论文 Introduction 进行评审。

**论文背景**: 本文研究分布式 MoE (Mixture-of-Experts) 推理中 Expert Parallelism 场景下的 all-to-all 通信调度优化。核心创新点是利用跨层流量可预测性实现多层联合调度，并辅以 traffic shaping 和 urgency-aware 优先级队列。

**请按两个版本分别评审**:

---

**版本 A: 立题评审 (实验尚未完成)**

假设实验尚未进行，仅从方向可行性和论文叙述角度评审。请重点关注:
- 方向是否值得做（问题真实性、空白合理性、INFOCOM适配度）
- 核心创新是否有足够的技术深度（是否有形式化、是否只是经典方法套用）
- 实验设计是否充分（baseline选择是否合理、消融实验是否能回答核心问题）
- 当前叙述中有哪些潜在的致命缺陷需要提前解决

**版本 B: 事后评审 (实验符合预期)**

假设以下预期结果已在实验中实现:
- 端到端推理延迟相比 NCCL EP baseline 降低 15-25%，相比 FAST 降低 8-15%
- P99 tail latency 降低 20-35%
- GPU利用率标准差降低 30-50%
- 消融实验: 多层联合调度贡献总改善的 50-60%，traffic shaping 贡献 25-35%，优先级调度贡献 10-20%
- 跨层预测准确率 90-95%，预测误差下性能graceful degradation

请从以下维度分别打分 (1-5分) 并给出理由:
1. 问题真实性与重要性
2. 空白合理性
3. 技术深度 (核心贡献是否有足够的技术深度，而非经典方法套用)
4. 收益可信度 (claimed improvement是否有充分的实证或理论支撑)
5. INFOCOM适配度
6. 引言写作质量 (逻辑链条是否清晰、是否在前2段给出problem+insight、是否有quantitative preview)

请明确指出:
- 最强的一个 contribution 是哪个，为什么
- 最弱的一个 contribution 是哪个，为什么
- 是否存在"审稿人必杀问题"(即无论如何回答都无法过关的硬伤)

请给出总体判断: Strong Accept / Accept / Weak Accept / Borderline / Weak Reject / Reject

如果总体判断低于 Weak Accept，请具体说明需要补充什么证据或修改什么叙述才能提升到 Weak Accept 以上。

---

### 论文题目

Traffic-Aware Dispatch Scheduling for MoE Expert Parallel Inference

### Introduction

Mixture-of-Experts (MoE) 架构已成为大语言模型规模化扩展的主流范式。DeepSeek-V3 (671B参数)、Mixtral、OLMoE等代表性模型均采用MoE结构，在每次前向传播中仅激活少数专家子集，在保持推理计算量可控的同时实现参数规模的大幅扩展。在分布式部署中，Expert Parallelism (EP) 是承载MoE架构的标准并行策略——将不同专家分配至不同GPU，token通过路由网络动态选择目标专家后，经由两次all-to-all集合通信（dispatch与combine）完成token-expert匹配和结果汇聚。这一流程使得每个MoE层都引入两次all-to-all通信，其累计开销在端到端推理延迟中占据30%至50%以上，已成为制约MoE推理性能的核心瓶颈。

围绕all-to-all通信效率的优化，现有工作沿三个层面展开。传输层优化（DeepEP、NCCL EP、Hybrid-EP、TensorRT-LLM NVLink One-Sided AlltoAll）关注如何在给定流量模式下更高效地搬运数据，优化的是数据搬运的物理路径和传输机制，但将每次all-to-all的流量矩阵视为给定输入，系统只能被动地"接到多少就搬多少"。调度层优化（FAST NSDI'26、Aurora IEEE'24、Lina ATC'23）关注在单次all-to-all内部如何安排通信顺序以避开网络瓶颈，其优化范围局限于单次操作内部，不涉及跨请求的流量协调——即使将FAST扩展到多层场景，它仍然是在每层流量产生后做reactive调度，而非在流量产生前做proactive规划。通信量缩减（s-MoE ICML'25、ExFlow、MoEShard）关注从根源减少需要传输的数据量，但与通信调度正交——即使通信总量已最小化，剩余的all-to-all通信仍可能存在时间分布不均和优先级不分的问题。上述三个层面的工作共享一个根本假设：每次all-to-all操作是独立的通信事件，流量模式是给定的输入，优化发生在流量确定之后。

本文挑战这一假设。我们通过对真实MoE推理流量的系统性量化分析，揭示了两个被忽视的结构性特征。

第一个特征是极端的流量skew。我们对OLMoE-1B模型进行了覆盖32个推理窗口和4层MoE层的量化分析（1024条采样记录），发现expert级别的激活次数差异达24倍，4-GPU EP配置下87%的评估组呈现3倍以上的流量倾斜且24%存在GPU完全零流量的情况，8-GPU配置下98%的评估组中至少有一个GPU处于完全空闲状态。这种极端skew不是偶发的边界条件，而是MoE稀疏路由的固有产物，意味着通信资源在大部分推理步骤中处于严重错配状态。

第二个特征是跨层流量的可预测性。这里的“可预测”不能再用相邻层 top-k overlap 近似，必须按 Fate 风格定义：用第 `L_i` 层的 gate 输入 hidden state 直接送入第 `L_{i+1}` 层 gate，测 `predicted_topK` 与 `actual_topK` 的 prefetch accuracy，并辅以 gate-input cosine similarity。当前 OLMoE-0924 的 32 prompt 批量 smoke 已显示：虽然跨层 top-k overlap 很低，但 Fate-style prefetch accuracy 在大量相邻层对上仍然较高，因此“低相似性 ≠ 不可预测”。

基于上述两个观察，我们提出RouterSENSE，一套应用层的流量感知dispatch调度框架。RouterSENSE的核心创新在于跨层预测驱动的多层联合调度，我们将该问题形式化为：给定K层MoE层的dispatch流量矩阵序列{T_1, T_2, ..., T_K}（其中T_l[i][j]表示第l层从GPU i发往GPU j的token数量），以及各GPU间的网络带宽约束，求解一个多层release plan R = {R_1, ..., R_K}，其中每层的release plan R_l指定该层各batch的发送顺序与发送时机，目标是在满足层间数据依赖约束（第l+1层的dispatch必须在第l层combine完成后才能开始）的前提下，最小化K层的总通信完成时间（sum of per-layer makespans）。由于层间数据依赖约束排除了pipeline overlap的可能，端到端延迟严格等于各层计算时间与通信时间之和，而各层计算时间为模型固有常量，因此最小化sum of per-layer makespans等价于最小化端到端推理延迟。这一问题等价于带前瞻窗口的非抢占式多机调度问题，我们通过从two-machine flowshop scheduling问题的归约证明其在K≥2时是NP-hard的，并提出一个基于前瞻式贪心的高效启发式算法，其决策开销为O(KG)（G为GPU数量），远低于单层all-to-all通信的毫秒级延迟，因此调度决策本身的开销可忽略不计。实验通过与oracle upper bound的对比验证该启发式算法在真实流量下的解质量。在此基础上，RouterSENSE进一步引入两个互补机制：(1) 基于实时积压状态的traffic shaping——不同于传统的token bucket或leaky bucket仅对单一流量流进行速率限制，RouterSENSE的shaping是多目的地感知的，根据各接收端GPU的实时积压状态动态分配发送带宽，在拥塞链路主动降速的同时在空闲链路加速发送，从而实现全局流量均衡而非单流限速；(2) 基于urgency-aware的优先级队列，利用推理运行时免费可得的结构性信号（topK rank、layer depth、expert load）区分dispatch的紧急度，确保关键路径上的dispatch优先完成。三个机制协同工作：跨层预测提供前瞻信息，多层联合调度决定各层怎么发整体最优，优先级调度决定同一层内谁先发。RouterSENSE不修改底层通信协议和传输机制，可与NCCL EP、FAST等传输层方案正交叠加。当跨层预测存在误差时，RouterSENSE的调度策略具备graceful degradation特性——错误的预测仅导致release plan退化为per-layer greedy，而不会产生比不做预测更差的结果。

我们在多节点多GPU集群上进行了系统性实验评估。实验结果表明，RouterSENSE相比NCCL EP baseline实现了X%的端到端推理延迟降低和X%的P99 tail latency降低，GPU利用率标准差降低X%。消融实验进一步证实，跨层预测驱动的多层联合调度贡献了总改善的X%，traffic shaping在高并发场景下贡献了X%，优先级调度作为增强模块贡献了X%。我们还构建了oracle upper bound（完美跨层预测下的全局最优解），实验表明oracle相对per-layer greedy的改善上限为X%，RouterSENSE在90-95%预测准确率下可捕获其中X%的改善。这些结果验证了应用层调度在不改变底层通信库的前提下即可显著改善MoE推理性能。

---

### 预期效果 (供评审参考)

以下两档预期用于评审者判断实验完成后的论文质量。

**档一: 保守预期 (跨机 Ethernet 场景, 中等并发)**

在跨机 Ethernet 配置下（2节点×4GPU），中等并发度 (8-16 concurrent requests) 场景下:
- 端到端推理延迟相比 NCCL EP baseline 降低 10-15%，相比 FAST 降低 5-10%
- P99 tail latency 降低 15-25%
- GPU利用率标准差降低 20-30%
- 消融实验: 多层联合调度贡献总改善的 40-50%，traffic shaping 贡献 30-40%，优先级调度贡献 15-25%

这一档的结果足以支撑 Weak Accept，但需要辅以良好的理论分析和跨层预测的 top-K hit rate 证据来增强技术深度的说服力。

**档二: 乐观预期 (跨机 Ethernet 场景, 高并发 burst)**

在高并发 burst 场景下 (32+ concurrent requests, 明显的流量突发):
- 端到端推理延迟相比 NCCL EP baseline 降低 20-30%，相比 FAST 降低 10-20%
- P99 tail latency 降低 30-50%
- GPU利用率标准差降低 40-60%
- 消融实验: 多层联合调度贡献总改善的 50-60%，traffic shaping 在高并发下贡献显著提升至 25-35%，优先级调度贡献 10-20%
- 跨层预测的 top-K hit rate ≥ 0.50，证明模式级预测可行

这一档的结果足以支撑 Accept，特别是如果 traffic shaping 在高并发下展现出显著的排队延迟降低，审稿人对其“经典方法套用”的质疑会被实验数据化解。

---

## 设计评审

### 评审提示词

你是一位分布式系统与计算机网络方向的资深审稿人。请对以下 RouterSENSE 系统设计方案进行评审。

评审重点：
1. Go/No-Go 决策门的判定指标是否合理，是否存在数据粒度与指标不匹配的风险
2. 四层模块架构是否与现有代码结构一致，接口设计是否合理
3. 关键数据流设计是否可行，调度器的插入点是否正确
4. 消融实验矩阵是否完整，是否缺少关键对照实验（如"真实流量多层联合调度" vs "跨层预测多层联合调度"）
5. 术语精确性（per-layer greedy vs 多层联合调度的本质区别）

请明确指出每个问题的严重程度（致命/高/中/低），并给出具体的修正建议。

### 一、设计目标与约束

RouterSENSE 系统实现的目标是：在不修改 NCCL 等底层通信库的前提下，在应用层实现三个协同的 dispatch 调度机制——跨层预测驱动的多层联合调度、多目的地感知 traffic shaping、urgency-aware 优先级队列——并通过消融实验量化各机制的独立贡献。

核心约束条件：(1) 自建 runtime，不使用 DeepSpeed-MoE/Megatron 等框架，因为框架锁死调度逻辑；(2) 通信底层直接调用 NCCL all_to_all_single，不自建；(3) 所有调度决策必须在 dispatch 开始前完成，决策开销远低于毫秒级 all-to-all 延迟；(4) 实验在真实多节点多 GPU 集群上运行，不使用仿真。

### 二、Go/No-Go 决策门

在进入系统实现之前，必须先通过两个决策门。这两个实验均可用 POC1 trace 数据离线完成，不需要实机多 GPU 环境。

**Gate 1：跨层相关性验证**

在 OLMoE-1B 上跑完整 forward pass，提取所有 MoE 层的 router logits，计算相邻层 expert 选择的 top-K hit rate（而非 Jaccard——单 token + 64 experts + topK=8 的场景下 Jaccard 会产生极低值 ≈ 0.07，不能反映 batch 级别的跨层可预测性）。

同时计算 batch 级别流量分布的 rank correlation。具体定义：对每个 src GPU，取其 outbound traffic vector（T[l][src_gpu][*]，长度为 G），计算相邻层的 Spearman rank correlation。对所有 src GPU 取平均后得到单层的 rank correlation。这一指标直接衡量调度输入的相似度——如果 rank correlation 高，说明相邻层的最优调度方案相似，多层联合调度的收益空间大。

hit rate 与 rank correlation 的关联分析：当 hit rate = 0.50（共享 4 个 expert）时，对应的 rank correlation 取决于这 4 个共享 expert 的流量占比是否一致。如果共享 expert 在两层都是头部 expert，则 rank correlation 可能仍然较高；如果共享 expert 的流量占比在两层差异大，则 rank correlation 低。Gate 1 需同时报告这两个指标，如果 hit rate ≥ 0.50 但 rank correlation < 0.30，说明跨层 expert 选择有重叠但流量分布差异大，多层联合调度的实际收益可能有限。判定标准：相邻层 top-K hit rate ≥ 0.50 → 通过；0.30–0.50 → 灰色地带；< 0.30 → 跨层预测不可行，D11 方向放弃，全力做 D10+D20。

Gate 1 与 Fate 的差异：Fate 在 Qwen-MoE、DeepSeek-MoE 等模型上验证了 97% 准确率，但 OLMoE 的路由器架构不同（不同的 top-K 值、不同的 expert 数量），必须在目标模型上直接验证。

**Gate 2：Oracle vs Greedy 上界实验**

构建一个纯离线模拟器，输入为多层流量矩阵序列 {T_1, T_2, ..., T_K}（从 POC1 的 1024 条 ablation 数据中提取），实现两种调度策略并对比：

- Per-layer Greedy：每层独立做局部贪心调度（LPT 类策略，非 FIFO，baseline）
- Multi-layer Oracle：拥有所有层的完美流量信息，通过精确求解或高精度近似求解全局最优 release plan

判定标准：Oracle 相对 Greedy 的 makespan 改善 ≥ 15% → 方向立住；5–15% → 灰色地带，需分析在哪些流量模式下改善显著；< 5% → 多层联合调度收益不足，需 pivot。

Oracle 实现策略：由于问题在 K≥2 时 NP-hard，精确求解使用整数规划（小规模 instance，如 G=4、K=4），保证在小规模下拿到精确最优解；大规模下使用高精度 heuristic（如 simulated annealing with long cooling）作为 oracle proxy。

### 三、模块架构

系统采用四层架构，与现有代码分层保持一致。

**分析层（analysis/）**——纯离线，不依赖 NCCL

该层从 POC1 trace 数据中提取流量矩阵并验证跨层相关性。输出为 traffic matrix 序列文件和跨层相关性报告。主要模块：

- `traffic_matrix.py`：从 OLMoE router trace（`olmoe_router_trace.py` 已实现的 `RouterTraceRecord`）中提取 per-layer per-step traffic matrix T[l][i][j]，其中 i/j 为 GPU rank。输出为结构化的 JSON 格式，供后续模拟器直接消费。
- `cross_layer_analyzer.py`：计算相邻层 expert 选择的 top-K hit rate、多层衰减曲线。输出可视化报告。
- `oracle_simulator.py`：读取 traffic matrix 序列，实现 Greedy 和 Oracle 两种调度策略，输出 makespan 对比和改善百分比。

**策略层（core/）**——模型无关的调度算法

该层实现三个调度策略，均继承自现有的 `Scheduler` 基类（`core/scheduler.py`）。三个调度器直接放在 `core/` 下（与现有 `scheduler.py` 同级），避免增加不必要的目录层级。现有 `Scheduler` 仅有 `plan(buckets, strategy) -> SchedulerDecision` 接口，需扩展为支持多层输入：

```python
class Scheduler:
    supports_multi_layer: bool = False  # 子类声明是否支持多层

    def plan(self, buckets, strategy) -> SchedulerDecision:
        """现有接口，作为 plan_single_layer 的 wrapper 保留向后兼容"""

    def plan_single_layer(self, traffic_matrix: dict, ...) -> list[int]:
        """Per-layer greedy: 现有行为"""

    def plan_multi_layer(self, traffic_matrices: list[dict], ...) -> MultiLayerReleasePlan:
        """多层联合调度: D11 核心"""
```

新增数据结构：

```python
@dataclass
class MultiLayerReleasePlan:
    layer_plans: list[LayerReleasePlan]  # 每层的 batch 顺序和发送时机
    layer_earliest_start: list[float]    # 每层 dispatch 的最早可开始时间（层间依赖约束，运行时计算）
```

三个调度器：

- `MultiLayerScheduler`（D11）：输入多层 traffic matrix，输出 MultiLayerReleasePlan。核心算法为前瞻式贪心：从第 1 层开始，每层调度时考虑后续 K-1 层的预测流量，避免当前层的 greedy 决策导致后续层被挤占。O(KG) 复杂度。
- `TrafficShapingScheduler`（D10）：多目的地感知的 pacing 策略。维护每个 destination rank 的积压状态，在拥塞链路延迟发送、在空闲链路加速发送。接口上接受 `BacklogState`（各 GPU 当前 pending bytes）作为额外输入。
- `PriorityScheduler`（D20）：基于 urgency 信号的优先级排序。信号统一为 cost 信号（值越大越后发）：`cost = α * rank_in_topK + β * (1 / layer_depth) + γ * expert_load`，权重通过网格搜索确定。

**运行时层（runtime/distributed_ep/）**——与现有代码集成

现有运行时的关键路径为：`olmoe_adapter.build_dispatch_plan_from_trace()` → `CollectiveOps.dispatch()` → `WorkerLoop.record_plan()`。调度器在此路径上的插入点为 dispatch 之前，即在 `build_dispatch_plan` 产出 `DispatchPlan` 之后、`CollectiveOps.dispatch` 之前，插入调度决策。

需要扩展的现有模块：

- `WorkerLoop`：新增 backlog 状态追踪——每次 dispatch + combine 全部完成后更新各 rank 的 pending bytes，供 TrafficShapingScheduler 消费。注意 backlog 状态的时间语义是“稳态”（所有 combine 完成后），而非 dispatch 发出后的瞬态，避免包含尚未 combine 完成的混合信息。
- `CollectiveOps`：新增 timing 记录——dispatch/combine 的开始和结束时间戳，供实验评估使用。
- `DistributedManifest`：新增 `scheduling_metadata`，记录使用的调度策略和参数。

**适配器层（adapter/）**——模型相关

现有 `olmoe_adapter.py` 已实现 OLMoE 的 dispatch plan 构建和本地 expert 执行。新增需求：

- 跨层预测模块：在 Layer l 的 forward pass 完成、gate logits 可用后，立即预测 Layer l+1, l+2, ... 的 traffic matrix。预测方法为 top-K argmax（与 Fate 一致）。预测计算与后续 dispatch 通信重叠执行，不阻塞 dispatch。
- 全层 trace 采集：现有 `olmoe_router_trace.py` 的 `collect_olmoe_router_trace` 只采集单层，需扩展为采集所有 MoE 层的 router logits。

### 四、关键数据流

推理流程中的数据流经路径：

```
forward pass Layer l (gate 计算 + expert 计算)
  → gate logits 在 Layer l 的 forward pass 结束时可用（expert forward 之前）
  → [预测] 用 Layer l logits 预测 Layer l+1..l+K-1 traffic matrix（与 dispatch 通信重叠执行）
  → [调度] 在 Layer l 的 dispatch 开始之前完成所有调度计算：
      ① MultiLayerScheduler 产出 Layer l..l+K-1 的 release plan（各层 batch 发送顺序和时机）
      ② TrafficShapingScheduler 在该 plan 的约束下调整每层内部的发送节奏（pacing）
      ③ PriorityScheduler 在同一发送节奏内对 batch 按 urgency 排序
  → NCCL all_to_all dispatch (Layer l)
  → expert forward
  → NCCL all_to_all combine
  → [更新] WorkerLoop 更新 backlog 状态
  → forward pass Layer l+1 (repeat)
```

三个调度器的组合方式：MultiLayerScheduler 先决定多层 plan（宏观编排）→ TrafficShapingScheduler 在该 plan 的约束下调整每层内部的发送节奏（微观 pacing）→ PriorityScheduler 在同一发送节奏内对 batch 内部排序。三者的输出维度不同但操作对象层次分明，不存在冲突。

关键设计决策：调度器拿到的流量矩阵中，Layer l 是精确的（来自当前 gate 输出），Layer l+1..l+K-1 是预测的。随着 forward pass 推进，预测窗口向前滑动，每层的精确流量矩阵逐步替换预测矩阵。

延迟预算分析：gate logits → traffic matrix 构建 → 多层调度决策 → pacing 调整的总延迟必须在 dispatch 可容忍的等待时间内完成。O(KG) 的调度计算本身是轻量操作（K=4, G=8 时仅 32 次操作），但 traffic matrix 构建涉及 token-expert 映射和聚合，需确保总开销远低于毫秒级 all-to-all 延迟（实机实验中将直接测量）。

re-plan 策略：当 Layer l+1 的实际流量与预测偏差大时，基于错误预测做出的 Layer l+2..l+K-1 的调度计划需要在该层 dispatch 前重新计算。由于每层的精确流量矩阵在 forward pass 推进时自动替换预测矩阵，re-plan 是滑动窗口的自然行为而非额外开销。

### 五、实验设计

**Baseline 配置**

| 名称 | 描述 | 来源 |
|---|---|---|
| NCCL EP | NVIDIA 标准 EP 实现，FIFO 调度 | 业界标准 |
| FAST | 单次 all-to-all 内部通信顺序优化 | NSDI'26 |
| Per-layer Greedy | 每层独立局部贪心调度（LPT 类策略，非 FIFO，不考虑跨层影响） | RouterSENSE baseline |
| DeepEP | 传输层优化（NVLink 场景） | 补充 baseline |

**消融实验矩阵**

| 实验 | D10 | D11 | D20 | 验证目标 |
|---|---|---|---|---|
| Exp 0 (baseline) | - | - | - | 基准线 |
| Exp 1 | ✅ | - | - | traffic shaping 独立贡献 |
| Exp 1.5 | - | ✅ | - | 多层联合调度独立贡献（不加 shaping/priority） |
| Exp 2 | ✅ | ✅ | - | 跨层预测增量 |
| Exp 2.5 | ✅ | ✅* | - | 真实流量多层联合调度（无预测） |
| Exp 3 | ✅ | - | ✅ | 优先级增量 |
| Exp 4 | ✅ | ✅ | ✅ | 完整方案 |
| Exp 5 (vs FAST) | ✅ | ✅ | ✅ | 与 transport 层方案对比 |
| Exp 6 (oracle) | - | ✅* | - | 启发式 vs 最优解 gap |
| Exp 7 (正交性) | ✅ | ✅ | ✅ | RouterSENSE + FAST 叠加，验证正交性声明 |

Exp 2.5 中的 D11* 使用当前层真实流量矩阵（非预测）做多层联合调度。这是一个纯离线实验：事后用所有层的真实流量矩阵做 oracle 式调度，回答“改善来自联合优化本身还是跨层预测”这一关键归因问题。注意与 Exp 6 的区别：Exp 2.5 用真实流量但求解的是启发式解（非全局最优），Exp 6 用完美预测流量但求解的是全局最优解。如果 Exp 2.5 改善 20% 而 Exp 2 改善 18%，说明改善主要来自联合优化，预测只是锦上添花；如果前者 20% 后者 5%，说明预测是核心。

Exp 6 中的 D11* 使用 oracle 完美预测替代实际预测，验证启发式算法的解质量。

Exp 7 将 RouterSENSE 与 FAST 同时启用，验证两者叠加时的效果是否等于各自单独效果之和（正交性声明的实验验证）。

**敏感度分析**

补充跨层预测准确率 vs 调度质量的参数曲线：当预测准确率从 90% 变化到 99% 时，多层联合调度的改善幅度如何变化。这为“预测误差对性能的影响”提供定量回答。

**评估维度**

端到端推理延迟（avg / P50 / P99）、各层 all-to-all 通信延迟、GPU 利用率标准差（fairness 指标）、调度决策开销（μs 级）。

---

## POC-line1-i: 方向验证

### 评审提示词

你是一位分布式系统方向的资深审稿人。请对以下 POC-line1-i 任务书进行评审。

评审重点：
1. 数据粒度是否适合验证目标（POC1 ablation 是单 token 数据，与 batch 级验证目标的匹配度）
2. 判定指标是否合理（hit rate vs Jaccard 的选择）
3. 流量矩阵聚合策略是否合理
4. Oracle 模拟器的建模细节是否正确
5. 局限性声明是否完整

请明确指出每个问题的严重程度（致命/高/中/低），并给出具体的修正建议。

### 目标

在不跑实机多 GPU 实验的前提下，用 POC1 离线数据回答两个问题：

1. OLMoE-1B 的跨层 expert 选择重叠度是否足够高？（Gate 1）
2. 多层联合调度相对 per-layer greedy 的收益上界有多大？（Gate 2）

### 可用数据

**Ablation checkpoint (1024 条)**
- 路径: `archive/poc1-20260629-local-prompts-0924/outputs/local_prompt_poc_0924/ablation_checkpoint.jsonl`
- 模型: OLMoE-1B-7B-0924（64 experts, topK=8）
- 4 个 MoE 层: layer 0, 5, 10, 15
- 32 个 windows, 每 window 取 token_pos=23 的 expert ranking
- 每条记录: window_id, layer_id, expert_id, topk_rank (0-7), router_logit, router_probability

**Router trace (3458 条)**
- 路径: `archive/poc1-20260629-local-prompts-0924/outputs/local_prompt_trace_0924/router_trace.json`
- 8 个 windows, 3 个 MoE 层（不全）
- 每条记录包含完整 expert ranking（不仅是 top-8）

### 任务拆解

**Task 1: 跨层 hit rate 分析 (Gate 1)**

输入: ablation_checkpoint.jsonl 的 1024 条记录

做法:
- 对每个 window，提取每层的 expert 选择集合（topK rank 0-7 共 8 个 expert）
- 主指标: top-K hit rate: accuracy(L_i → L_{i+1}) = |S_i ∩ S_{i+1}| / K（不计算 Jaccard——单 token + 64 experts 场景下 Jaccard ≈ 0.07，不代表方向不可行）
- 补充指标: batch 级别流量分布的 Spearman rank correlation（对每个 src GPU 的 outbound traffic vector 分别计算，取平均）
- 多层衰减: accuracy(L_i → L_{i+k}) for k=1,2,3
- 按 window 和 layer pair 聚合统计

输出:
- 相邻层平均 top-K hit rate、中位数、分布
- top-2 / top-4 / top-8 的 hit rate
- 多层衰减曲线
- 可视化: heatmap + histogram

判定:
- 相邻层 top-K hit rate ≥ 0.50 → Gate 1 通过
- 0.30–0.50 → 灰色地带
- < 0.30 → Gate 1 不通过

代码: `experiments/poc_line1/cross_layer_analysis.py` ~100行

**Task 2: 流量矩阵提取**

输入: ablation_checkpoint.jsonl

做法:
- 假设 4-GPU EP（round-robin placement: expert_i → GPU_{i % 4}）
- src_gpu 确定方式：假设 token 在 dispatch 前均匀分布在所有 GPU 上（每 GPU 持有 batch 中的部分 token）。在 POC1 单 token 场景下，每 GPU 各持 1/N 的 token（N=4 时每个 GPU 0.25 个 token）；在聚合 batch 场景下，batch_size 个 token 均匀分配到 4 个 GPU（batch_size=8 时每 GPU 2 个 token）。这个假设与真实 EP 场景一致（forward pass 后每个 GPU 持有该 GPU 上的 token 子集）
- 对每个 window 和每层，构建 traffic matrix T[l][src_gpu][dst_gpu]:
  - 每个 token 的 top-2 expert 选择映射到对应的 dst GPU
  - 统计每个 src_gpu 到每个 dst_gpu 的 token 数量
  - 由于 POC1 是单 token 分析（token_pos=23），每个 window 只有 1 个 token，所以 T[l] 是非常稀疏的矩阵
  - 为了模拟更真实的批量场景，将多个 window 聚合为一个 batch

聚合策略（两种都做，作为分析维度）:
- 方案 A（顺序聚合）: window 0-7 → batch 0, window 8-15 → batch 1, window 16-23 → batch 2, window 24-31 → batch 3。保留时间局部性，更接近真实推理场景
- 方案 B（随机聚合）: 随机选择 8 个 window 组成 batch，重复多次。消除时间相关性，测试泛化性

聚合后 token 数量: 8 tokens per batch，4 GPU EP 下每 token 最多发往 2 个 GPU（topK=8 expert 分布在 64/4=16 expert/GPU），traffic matrix 非零元素足够填充 4×4=16 个元素，产生有意义的调度空间。

聚合规模参数 sweep: 测试 batch_size = {4, 8, 16, 32} 下 Oracle vs Greedy 的改善幅度，报告改善幅度随 batch size 的变化趋势。稀疏 matrix 下 Greedy 和 Oracle 差异小（可行解空间小），密集 matrix 下差异大。这一步用于确定调度优化收益与 batch size 的关系。对于 batch_size=32（用尽全部 window，只有 1 个 batch），采用 bootstrap resampling：从 32 个 window 中随机采样（有放回）构造多个 batch_size=32 的 batch，报告改善幅度的 95% 置信区间。

输出: per-batch per-layer traffic matrix (JSON 格式)，包含顺序和随机两种聚合结果

代码: `experiments/poc_line1/traffic_matrix_extractor.py` ~80行

**Task 3: Oracle vs Greedy 模拟器 (Gate 2)**

输入: Task 2 产出的 traffic matrix 序列

做法:
- 实现 `GreedyScheduler`: 每层独立局部贪心调度（不考虑跨层影响），最小化本层 makespan
- 实现 `OracleScheduler`: 拥有所有层的完美流量信息，通过 ILP（scipy.optimize 或 PuLP）求解全局最优 release plan
  - 小规模 (G=4, K=4): 精确求解
  - 如果 ILP 求解太慢，用 simulated annealing 作为 oracle proxy（需先在小规模 instance 上与 ILP 精确解对比，确认 SA 的 gap < 2%）
- 调度模型（详细定义见 poc_line1_i_task.md）:
  - 核心约束: per-GPU half-duplex（每个 GPU 同一时刻只能参与一个传输），不同 GPU 可以并行传输
  - 层间依赖（放宽为 per-GPU 粒度）: Layer l+1 中涉及 GPU i 的 chunk 必须等 Layer l 中涉及 GPU i 的 combine 完成后才能开始（而非全局等待——全局等待会导致 Oracle = Greedy）
  - combine 时间建模为 dispatch 转置 × combine_scale_factor
  - 目标: 最小化所有层所有 chunk 完成的总时间

输出:
- 每个 batch 的 greedy makespan vs oracle makespan
- 改善百分比: (greedy - oracle) / greedy × 100%
- 统计分布和 worst case / best case
- **skew 放大分析**：在 traffic matrix 上人为增加 skew 程度（通过 weighted sampling 让 hot expert 更 hot，skew 系数从 1x 到 5x），观察 Oracle vs Greedy 改善幅度随 skew 程度的变化曲线。如果改善幅度随 skew 增大而显著增大，则可以论证“真实场景 skew 更极端 → 改善更大”，即使 POC 数据本身 skew 不够极端也能支撑方向可行性

判定:
- Oracle 改善 ≥ 15% → Gate 2 通过
- 5–15% → 灰色地带
- < 5% → 方向需 pivot

代码: `experiments/poc_line1/oracle_simulator.py` ~200行

### 目录结构

```
experiments/poc_line1/
├── __init__.py
├── cross_layer_analysis.py      # Task 1: hit rate 分析
├── traffic_matrix_extractor.py  # Task 2: 流量矩阵提取
├── oracle_simulator.py          # Task 3: Oracle vs Greedy
└── run_all.py                   # 统一入口脚本
```

### 执行顺序

Task 1 → Task 2 → Task 3，串行执行。Task 1 如果不通过，Task 3 仍应完成（即使跨层预测不可行，多层联合调度的收益上界分析仍然有价值——可以用真实的当前层流量做多层联合调度，不需要预测）。

### 局限性

这个 POC 的已知局限性：
1. POC1 ablation 是单 token 分析（每 window 只取 token_pos=23），不是完整 batch 的流量矩阵。人工聚合的 batch（跨 window 拼接）与真实 batch（同一请求中多 token 共享 prompt context，expert 选择具有语义相关性）的流量模式有本质差异。Gate 2 的定位应为“可行性初探”而非严格的方向验证
2. 假设 round-robin expert placement，不是生产环境的实际放置。不同的 placement 策略会根本性地改变 traffic matrix 的结构——如果生产环境使用 load-balanced 或 affinity-based placement，Gate 2 的结论可能完全不同
3. 模拟网络传输时间为 batch 大小的线性函数，未建模 NCCL 实际行为（线性假设在跨机 Ethernet 场景下大致成立，NVLink/InfiniBand 下不成立）
4. 层间 combine 时间简化为 dispatch_time × ratio，未建模 combine 的独立流量模式
5. Gate 1 验证的是 token 级别的跨层 expert 选择重叠度，而 RouterSENSE 实际利用的是 batch 级别的跨层流量模式可预测性。即使单 token 的 expert 选择跨层差异大，batch 级别的流量分布仍可能高度相似——这一差距需要在后续实机实验的 batch-level trace 中弥补
6. OLMoE-1B 是 1B 参数的小模型，其路由行为可能与更大规模模型（DeepSeek-V3 671B、Mixtral 8x7B）有本质差异。Gate 1/2 的结论可能不适用于更大模型
7. 每 window 只取 token_pos=23（单位置取样），而不同位置的 token（首 token vs 中间 token vs 末尾 token）的 expert 选择模式可能差异显著。Gate 1 的结论可能只适用于特定位置的 token
8. POC 数据来自单请求推理（单条 prompt 的 forward pass），而实际推理服务是多请求并发的。多请求并发时，不同请求的 token 在 all-to-all 中混合，traffic matrix 的 skew 模式可能与单请求场景截然不同

这些局限性在 Gate 1/2 通过后的实机实验中解决。
