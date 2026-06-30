# POC-line1-i: 跨层预测与多层联合调度离线验证

## 一、变更说明

本文档是对 `new-directions.md` 中 POC-line1-i 任务书的**修订版**。基于立题评审、设计评审、POC 评审三方收敛意见，原任务书存在以下问题，本修订逐一修正并给出实现细节。

### 核心变更

| 维度 | 原版 | 修订版 | 变更原因 |
|---|---|---|---|
| trace 采集 | 单 token（token_pos=23） | 全序列所有 token position | 单 token 构建的 traffic matrix 极端稀疏，Gate 2 结论不可信 |
| MoE 层覆盖 | ablation: 4层 (0,5,10,15)；trace: 3层 (0,8,15) | 全部 router-active 层（当前实测 OLMoE-1B-7B-0924-Instruct 返回 16 个 router layers） | 旧的 4 层假设与真实模型不符，需按 16 层重估 |
| Gate 1 指标 | top-2 Jaccard ≥ 0.80 | top-K hit rate ≥ 0.60（主）+ weighted hit rate（辅） | Jaccard 在 64 experts + topK=8 下 ≈ 0.07，不 informative |
| traffic matrix | 跨 window 人工聚合 | 同 prompt 内多 token 真实聚合 + 跨 window 聚合对照 | 跨 window 聚合不反映真实 batch 语义相关性 |
| expert placement | 仅 round-robin | round-robin + skewed（hot expert 集中放置） | round-robin 人为平滑 skew，压缩 Oracle vs Greedy 差异 |
| makespan 建模 | 未定义 | per-destination 调度粒度 + shared link max 模型 | 原建模模糊，影响 Gate 2 结论 |
| 补充实验 | 无 | Exp 2.5b（有预测无联合调度）+ 模型泛化 trace | 评审要求归因更精细 + 泛化性验证 |

### 新增目标（评审收敛后追加）

1. **多 token position 分析**：Gate 1 不能只验证一个位置，需覆盖首/中/尾 token 的跨层相关性差异
2. **batch size sweep**：Gate 2 需报告 Oracle vs Greedy 改善幅度随 batch size 的变化趋势
3. **combine 时间建模**：combine 的流量矩阵建模为 dispatch 的转置，而非简单 ratio
4. **Per-layer Greedy 精确定义**：明确为 LPT (Longest Processing Time first)，配 2-layer 4-GPU 反例说明全局次优

---

## 二、整体架构

```
Phase 0: 全序列 Trace 采集（单 GPU，需模型）
  ↓ 产出: full_sequence_trace.jsonl
Phase 1: 跨层相关性分析（纯离线，不需 GPU）→ Gate 1 判定
  ↓ 产出: cross_layer_report/
Phase 2: 流量矩阵构建（纯离线）
  ↓ 产出: traffic_matrices/
Phase 3: Oracle vs Greedy 模拟（纯离线）→ Gate 2 判定
  ↓ 产出: oracle_report/
Phase 4: 汇总报告 + Go/No-Go 决策
```

Phase 0 是唯一需要 GPU 和模型的阶段。Phase 1-4 全部离线计算，可在任意环境运行。

---

## 三、Phase 0: 全序列 Trace 采集

### 目标

在 OLMoE-1B-7B-0924 上跑完整 forward pass，采集**所有 MoE 层、所有 token position** 的 router logits 和 top-K expert 选择。

### 输入

- 模型: `allenai/OLMoE-1B-7B-0924-Instruct`（64 experts, topK=8, 4 个 MoE 层位于 layer 0/5/10/15）
- prompts: 复用 POC1 的 32 条 theory prompts（路径: `archive/poc1-20260629-local-prompts-0924/data/theory_prompts_50.jsonl`，取前 32 条）
- 环境: 单 GPU，bf16 精度

### 实现要点

修改或扩展现有 `src/routesense/trace/olmoe_router_trace.py` 中的 `collect_olmoe_router_trace` 函数。当前实现只采集 `layer_path` 指定的单层，需要新增一个全层采集函数：

```python
def collect_full_sequence_trace(
    model, tokenizer, text: str,
    *, request_id: str = "req-0",
    sample_id: str = "sample-0",
) -> dict:
    """采集所有 MoE 层、所有 token position 的 router trace。"""
    model.eval()
    encoded = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded, output_router_logits=True, return_dict=True, use_cache=False)
    router_logits_by_layer = outputs.router_logits  # list of K tensors, each shape [seq_len, num_experts]
    topk = model.config.num_experts_per_tok  # 8
    records = []
    for layer_idx, logits in enumerate(router_logits_by_layer):
        probs = torch.softmax(logits, dim=-1)        # [seq_len, 64]
        weights, experts = torch.topk(probs, k=topk, dim=-1)  # [seq_len, 8]
        for token_pos in range(logits.shape[0]):
            for rank_in_topk in range(topk):
                records.append({
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "token_position": token_pos,
                    "layer_id": layer_idx,
                    "expert_id": int(experts[token_pos, rank_in_topk]),
                    "topk_rank": rank_in_topk,
                    "routing_weight": float(weights[token_pos, rank_in_topk]),
                    "topk": topk,
                })
    return {
        "summary": {
            "request_id": request_id,
            "sample_id": sample_id,
            "moe_layer_count": len(router_logits_by_layer),
            "topk": topk,
            "token_count": int(router_logits_by_layer[0].shape[0]),
            "record_count": len(records),
        },
        "records": records,
    }
```

**关键注意事项**：
- 当前实现必须先运行 architecture probe。若 `router_logits` 返回 16 个 layer，则后续 Gate 1 / Gate 2 分析全部以 16 层为准，不再沿用 4 层假设。
- 32 条 prompt × 每条约 30-50 tokens × 4 MoE layers × topK=8 = 约 30,000-50,000 条记录。数据量可控。
- 输出文件用 JSONL 格式（一行一条记录），方便后续逐条处理。

### 输出

```
outputs/poc_line1/full_sequence_trace/
├── config.json          # 模型、prompts、采集参数
├── environment.json     # 运行环境快照
└── trace.jsonl          # 全量 trace 记录
```

每条记录 schema:
```json
{
  "request_id": "req-0",
  "sample_id": "prompt-0",
  "token_position": 7,
  "layer_id": 0,
  "expert_id": 48,
  "topk_rank": 0,
  "routing_weight": 0.1234,
  "topk": 8
}
```

---

## 四、Phase 1: 跨层相关性分析（Gate 1）

### 目标

回答：OLMoE-1B 的跨层 expert 选择重叠度是否足够高，使得跨层预测驱动的调度可行？

### 输入

Phase 0 产出的 `trace.jsonl`

### 分析项

#### 1.1 Token 级 hit rate 分析

对每个 (sample_id, token_position) 组合：

```
对于相邻层对 (L_i, L_{i+1}):
  S_i = {expert_id for topk_rank in 0..7 at layer L_i}
  S_{i+1} = {expert_id for topk_rank in 0..7 at layer L_{i+1}}
  hit_rate(K) = |S_i ∩ S_{i+1}| / K    # K = topK = 8
```

聚合维度：
- 按 layer pair 聚合: (0→5), (5→10), (10→15)
- 按 token position 分桶: first (pos < 10), middle (10 ≤ pos < seq_len-10), last (pos ≥ seq_len-10)
- 全局统计: mean, median, std, P25, P75

**多层衰减分析**：计算 hit_rate(L_i → L_{i+k}) for k=1,2,3，观察跨层衰减速度。

#### 1.2 Weighted hit rate 分析

```
weighted_hit_rate(L_i, L_{i+1}) = Σ_{e ∈ S_i ∩ S_{i+1}} (p_i(e) + p_{i+1}(e)) / 2
```

其中 p_i(e) 是 expert e 在 layer i 的 routing probability。这个指标区分"共享的 expert 是否是高概率 expert"。

#### 1.3 Batch 级流量分布 rank correlation

将同一 sample_id 下多个 token 聚合为 batch 级流量分布：

```
对于每个 sample_id 和每个 layer:
  gpu_flow[dst_gpu] = count of (token, topk_rank=0..1) 映射到 dst_gpu 的数量
  # 只取 topK 中 rank 0 和 1 的 expert（主要流量），映射方式见 Phase 2
```

然后计算相邻层 gpu_flow 向量的 Spearman rank correlation。

#### 1.4 可视化

- heatmap: layer pair × token position bucket 的 hit rate
- histogram: hit rate 分布（按 layer pair 分面）
- 衰减曲线: hit rate vs 跨层距离 k
- scatter: token-level hit rate vs batch-level rank correlation
- pairwise correlation matrix: 三个 Gate 1 指标（hit rate, weighted hit rate, rank correlation）两两之间的 Spearman/Pearson 相关系数，验证指标间是否提供互补信息

### 判定标准

| 指标 | 通过 | 灰色地带 | 不通过 |
|---|---|---|---|
| 相邻层平均 hit rate (K=8) | ≥ 0.60 | 0.40–0.60 | < 0.40 |
| 相邻层 weighted hit rate | ≥ 0.45 | 0.30–0.45 | < 0.30 |
| batch 级 rank correlation | ≥ 0.50 | 0.30–0.50 | < 0.30 |

三个指标中至少两个通过 → Gate 1 通过。

**阈值校准**：在正式运行前，先用 POC1 的 ablation 数据（1024 条，虽然只有 token_pos=23）做一次快速预跑（quick sanity check），校准各指标的分布，确认阈值是否需要微调。这比跑完 Phase 0 全量数据后才发现阈值不合理要安全得多。

### 与 Fate 的差异说明

Fate 在 Qwen-MoE/DeepSeek-MoE 上获得 97% 预测准确率，但这些模型的 router 架构与 OLMoE 不同（expert 数量、topK 值、路由器训练方式）。本分析是在目标模型上的直接验证，结论不可直接对比。

### 代码

`experiments/poc_line1/analysis/cross_layer_analyzer.py`，约 200 行。

---

## 五、Phase 2: 流量矩阵构建

### 目标

从 trace 数据构建多层 traffic matrix 序列，供 Phase 3 模拟器消费。

### Expert Placement 策略

实现两种 placement，对比 skew 对调度收益的影响：

**Placement A: Round-Robin**
```
expert_i → GPU_{i % G}
64 experts, G=4 GPUs → expert 0-15 → GPU 0, expert 16-31 → GPU 1, ...
```

**Placement B: Skewed (Hot Expert Concentration)**
```
模拟 hot expert 集中在少数 GPU 的场景。
将 POC1 ablation 数据中激活频率最高的 top-16 experts 映射到 GPU 0，
其余 48 experts 均匀分配到 GPU 1-3。
```

### Batch 构建策略

**策略 A: 同 prompt 真实 batch（主要）**

同一 sample_id 下的多个 token 组成一个 batch。这是真实推理场景：一个 prompt 的所有 token 同时参与 forward pass。

```
对于每个 sample_id:
  tokens = 该 prompt 的所有 token_position
  # 截断到 G 的倍数，避免分配不均引入非路由来源的 skew
  token_count = (len(tokens) // G) * G
  tokens = tokens[:token_count]
  对于每个 layer:
    T[layer][src_gpu][dst_gpu] = 从 src_gpu 发往 dst_gpu 的 token 数量
    其中:
      src_gpu = token_pos % G（均匀分配，与 expert 选择无关）
      dst_gpu = expert_id 映射到的 GPU（由 placement 策略决定）
```

**src_gpu 分配方式的明确声明与理由**：

在 EP 中，token 在 dispatch 前均匀分布在所有 GPU 上（每个 GPU 持有序列的一部分），与 expert 选择无关。因此 src_gpu = token_pos % G（均匀分配）是正确的默认选择。

如果 src_gpu 也按 expert affinity 分配（比如把经常选 GPU 0 上 expert 的 token 放在 GPU 0），skew 会被大幅缓解，但这不符合 EP 的实际行为。

**sensitivity analysis**：在结果报告中额外测试 src_gpu 按 expert affinity 分配的场景，观察 src_gpu 分配方式对 Oracle vs Greedy 改善幅度的影响。

**策略 B: 跨 prompt 聚合（对照）**

将不同 sample_id 的同一 token_position 聚合为一个 batch。这对应多请求并发场景。

```
对于每组 N 个 sample_id:
  对于每个 layer:
    T[layer][src_gpu][dst_gpu] = 聚合后的 token 流量
```

### Batch Size Sweep

batch_size ∈ {4, 8, 16, 32}。对于策略 A，batch_size = prompt 的 token 数量（截断或 padding）。对于策略 B，batch_size = 聚合的 prompt 数量。

### 输出

```
outputs/poc_line1/traffic_matrices/
├── round_robin/
│   ├── batch_4/
│   │   ├── batch_0.json    # {"batch_id": 0, "layers": [{"src": 0, "dst": 0, "count": 3}, ...]}
│   │   ├── batch_1.json
│   │   └── ...
│   ├── batch_8/
│   ├── batch_16/
│   └── batch_32/
├── skewed/
│   ├── batch_4/
│   └── ...
└── config.json              # placement 策略参数、batch size 配置
```

每个 batch JSON 的 traffic matrix schema:
```json
{
  "batch_id": 0,
  "sample_ids": ["prompt-0"],
  "token_count": 32,
  "placement": "round_robin",
  "num_gpus": 4,
  "layers": [
    {
      "layer_id": 0,
      "matrix": [[2, 3, 1, 2], [1, 4, 0, 3], [0, 2, 5, 1], [3, 1, 2, 2]],
      "row_labels": ["gpu_0", "gpu_1", "gpu_2", "gpu_3"],
      "col_labels": ["gpu_0", "gpu_1", "gpu_2", "gpu_3"]
    },
    {"layer_id": 5, "matrix": [...]},
    {"layer_id": 10, "matrix": [...]},
    {"layer_id": 15, "matrix": [...]}
  ]
}
```

`matrix[i][j]` = 从 GPU i 发往 GPU j 的 token 数量。

### 代码

`experiments/poc_line1/analysis/traffic_matrix_builder.py`，约 150 行。

---

## 六、Phase 3: Oracle vs Greedy 模拟器（Gate 2）

### 目标

量化多层联合调度相对 per-layer greedy 的收益上界。

### 调度模型

#### 调度粒度

**Per-destination GPU 调度**：每个 GPU 可以独立决定发往不同 destination GPU 的数据的发送顺序。这对应 NCCL all-to-all 的实际行为——每个 GPU 的 send buffer 中可以按 destination 分 chunk，chunk 之间可以重排序。

G=4 时，每层有 4 × 3 = 12 个独立的 send chunk（对角线元素 matrix[i][i] 不需要跨 GPU 传输）。

#### Makespan 建模

**Per-GPU 并发约束模型**（核心约束）：

每个 GPU 只有一个 NIC port，同一时刻只能向一个 destination 发送或从一个 source 接收（per-GPU half-duplex）。不同 GPU 可以同时向不同 destination 发送。

```
对于 4-GPU 拓扑 (每 GPU 在不同节点):
  每个 GPU 维护一个 available_time:
    GPU i 发送 chunk(i→j) 的时间窗口: [available[i], available[i] + size)
    GPU j 接收 chunk(i→j) 的时间窗口: [available[j], available[j] + size)
  约束: GPU i 和 GPU j 在该 chunk 传输期间都不能参与其他传输
  即: chunk 的开始时间 = max(available[i], available[j])
```

这个模型下单层内的调度顺序**直接影响**该层 makespan——因为 GPU 并发冲突导致不同的 chunk 排列产生不同的总时间。

**注意**：之前版本的 shared link 模型（所有 cross-node chunk 串行发送，makespan = 总流量 / 带宽）会导致单层 makespan 为常数（不依赖调度顺序），从而使 Oracle = Greedy，Gate 2 改善永远为 0%。修正为 per-GPU 并发约束后，调度问题有非平凡的解空间。

#### 调度问题形式化定义

```
输入:
  K 层 traffic matrix 序列 {T_1, ..., T_K}，每层 G×(G-1) 个 chunk
  combine matrix 序列 {C_1, ..., C_K}，其中 C_l = T_l^T × combine_scale_factor
  combine_scale_factor ∈ [0.5, 2.0]，默认 1.0

决策变量: 每个 chunk (l,i,j) 的开始时间 start_time[l][i][j] ∈ R+

约束:
  1. Per-GPU half-duplex: 同一 GPU 同一时刻不能同时参与多个传输
     对于同一 GPU g 参与的所有 chunk (发送或接收)，时间窗口不能重叠
  2. 放宽的层间依赖: Layer l+1 中涉及 GPU i 的 chunk 必须等 Layer l 中
     涉及 GPU i 的 combine 完成后才能开始（而非全局等待）
     start_time[l+1][i][j] ≥ combine_end_time[l][i]
     start_time[l+1][i][j] ≥ combine_end_time[l][j]

目标: minimize max over all chunks (start_time[l][i][j] + size[l][i][j])
```

**放宽层间依赖的关键意义**：全局等待（Layer l+1 必须等 Layer l 所有 chunk 完成）会导致总 makespan = 各层 makespan 之和（常数），Oracle 无法优于 Greedy。放宽为 per-GPU 依赖后，如果 Layer l 中 GPU 0 和 GPU 1 很早完成，它们可以立即开始 Layer l+1 的 dispatch，无需等其他 GPU 完成——这创造了非平凡的调度空间。

**这是一个 scheduling on parallel machines with precedence constraints 的问题，K≥2 时 NP-hard。**

#### Per-layer Greedy 实现

使用 LPT (Longest Processing Time first) 策略：每层内，按 chunk size 从大到小排序发送。层间使用 per-GPU 依赖传递。

```python
def greedy_schedule_single_layer(traffic_matrix: list[list[int]], num_gpus: int, 
                                 gpu_earliest_start: list[float] = None) -> tuple[float, list[float]]:
    """单层 LPT 调度，返回 (该层 makespan, 每 GPU 的完成时间)。
    
    gpu_earliest_start: 每个 GPU 的最早可用时间（来自上一层的 combine 完成时间）。
    返回的 gpu_finish_time 用于计算下一层的 gpu_earliest_start。
    """
    if gpu_earliest_start is None:
        gpu_earliest_start = [0.0] * num_gpus
    
    chunks = []
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j and traffic_matrix[i][j] > 0:
                chunks.append((traffic_matrix[i][j], i, j))
    chunks.sort(reverse=True)  # LPT: 最大的先发
    
    # 模拟调度: 维护每个 GPU 的可用时间
    gpu_available = list(gpu_earliest_start)  # 从上一层的完成时间开始
    for size, src, dst in chunks:
        start = max(gpu_available[src], gpu_available[dst])
        end = start + size  # 归一化: 1 token = 1 time unit
        gpu_available[src] = end
        gpu_available[dst] = end
    return max(gpu_available), gpu_available


def greedy_schedule_multi_layer(traffic_matrices: list[list[list[int]]], 
                                 combine_matrices: list[list[list[int]]],
                                 num_gpus: int) -> float:
    """多层 Per-layer Greedy: 每层独立 LPT，层间 per-GPU 依赖。"""
    gpu_earliest_start = [0.0] * num_gpus
    for l in range(len(traffic_matrices)):
        # dispatch
        dispatch_makespan, gpu_after_dispatch = greedy_schedule_single_layer(
            traffic_matrices[l], num_gpus, gpu_earliest_start)
        # combine (转置矩阵, per-GPU 依赖从 dispatch 完成时间开始)
        _, gpu_after_combine = greedy_schedule_single_layer(
            combine_matrices[l], num_gpus, gpu_after_dispatch)
        # 下一层的 earliest_start 是 combine 完成时间
        gpu_earliest_start = gpu_after_combine
    return max(gpu_earliest_start)
```

#### Oracle 实现

**小规模精确求解 (G=4, K=4)**：使用 ILP（`scipy.optimize.milp` 或 `PuLP`）。

```python
# ILP 建模要点
# 决策变量: start_time[l][c] ∈ R+，chunk c 在层 l 的开始时间
# 约束: 
#   (1) 同一 GPU 不重叠（per-GPU half-duplex）
#   (2) 放宽的层间依赖: start_time[l+1][c] ≥ combine_end_time[l][GPU(c)]
#       而非全局等待（start_time[l+1][c] ≥ max_end_time[l]）
# 目标: minimize max over all (start_time[l][c] + size[l][c])
```

如果 ILP 求解超过 60 秒，降级为 Simulated Annealing：
- 初始解: Greedy 解
- 邻域操作 1: 交换同一层内两个 chunk 的顺序
- 邻域操作 2: 跨层 chunk 交换——将 Layer l 的某个 chunk 和 Layer l+1 的某个 chunk 在时序上交换位置（前提是满足依赖约束），探索更大的解空间
- 温度: T_0 = 1000, cooling rate = 0.995, min_T = 0.01
- **校准**: 在 G=4, K=4 的小规模 instance 上与 ILP 精确解对比，确认 SA gap < 2%

#### Combine 时间建模

combine 的流量矩阵建模为 dispatch 的**转置**（expert output 发回 token 所在 GPU）：

```
combine_matrix[l][i][j] = dispatch_matrix[l][j][i] × combine_scale_factor
```

combine 阶段的调度也使用 LPT 策略（这是一个假设：真实 NCCL 中 combine 的调度顺序通常与 dispatch 相同或由内部控制，此处简化为独立 LPT 调度）。

`combine_scale_factor` 默认 1.0，在 sensitivity analysis 中测试 [0.5, 1.0, 1.5, 2.0] 的范围。物理含义：expert output 经过 down projection 后维度可能与 input token 不同。

层间依赖约束（per-GPU 粒度）：Layer l+1 中涉及 GPU i 的 dispatch 必须等 Layer l 中涉及 GPU i 的 combine 完成后才能开始。

```
combine_end_time[l][i] = GPU i 在 Layer l 的 combine 完成时间
# 不同 GPU 的 combine 完成时间可能不同
start_time[l+1][i][j] ≥ max(combine_end_time[l][i], combine_end_time[l][j])
```

### 实验配置

| 参数 | 值 |
|---|---|
| num_gpus (G) | 4 |
| num_layers (K) | 4 (layers 0, 5, 10, 15) |
| batch_sizes | [4, 8, 16, 32] |
| placement | round_robin, skewed |
| combine_model | transpose |
| combine_scale_factor | [0.5, 1.0, 1.5, 2.0] |
| batch_strategy | same_prompt, cross_prompt |

### 输出

```
outputs/poc_line1/oracle_report/
├── results.json          # 每个配置的 greedy_makespan, oracle_makespan, improvement_pct
├── summary.json          # 汇总统计
├── sensitivity/
│   ├── batch_size_effect.json      # improvement_pct vs batch_size
│   ├── placement_effect.json       # improvement_pct vs placement
│   ├── combine_scale_effect.json   # improvement_pct vs combine_scale_factor
│   └── src_gpu_effect.json         # improvement_pct vs src_gpu 分配方式
├── counterexample.json   # per-layer greedy 全局次优的 2-layer 4-GPU 反例（per-GPU 并发约束模型下）
└── gate2_decision.json   # {"passed": bool, "improvement_pct": float, "threshold": 15.0}
```

### 判定标准

| Oracle 改善 | 判定 |
|---|---|
| ≥ 15% | Gate 2 通过 |
| 5–15% | 灰色地带，分析在哪些流量模式下改善显著 |
| < 5% | 方向需 pivot |

### 2-Layer 4-GPU 反例

构造一个人工 traffic matrix 对，证明在 per-GPU half-duplex 约束下，per-layer greedy (LPT) 产生全局次优解。

**关键洞察**：在 per-GPU 并发约束下，LPT 对于单层问题近似最优（层内 makespan 几乎无差异）。Oracle 的优势来自**跨层协调**——Layer 1 的调度顺序影响各 GPU 的可用时间分布，从而影响后续层的调度效率。

```
2-Layer 4-GPU 反例（仅展示 dispatch 阶段，combine 类似处理）：

Layer 1 dispatch (3 chunks):
  GPU 0 → GPU 1: 7 tokens
  GPU 2 → GPU 0: 5 tokens
  GPU 1 → GPU 3: 5 tokens

Greedy (LPT): 按 size 降序 (7, 5, 5)
  (0→1, 7):  [0,7),   avail=[7,7,0,0]
  (2→0, 5):  start=max(avail[2],avail[0])=max(0,7)=7, end=12. avail=[12,7,12,0]
  (1→3, 5):  start=max(avail[1],avail[3])=max(7,0)=7, end=12. avail=[12,12,12,12]
  Layer 1: [12, 12, 12, 12]  ← 所有 GPU 同时完成

Oracle: 顺序 (5, 5, 7)  (先发两个不冲突的小 chunk)
  (2→0, 5):  [0,5),   avail=[5,0,5,0]
  (1→3, 5):  [0,5),   avail=[5,5,5,5]  ← GPU{2,0} 与 GPU{1,3} 不冲突，可并行！
  (0→1, 7):  start=max(avail[0],avail[1])=max(5,5)=5, end=12. avail=[12,12,5,5]
  Layer 1: [12, 12, 5, 5]  ← GPU 2,3 提前 7 个时间单位完成！

Layer 1 makespan 相同 (12)，但 GPU 可用时间分布不同：
  LPT: [12, 12, 12, 12]  — 所有 GPU 同时完成
  Alt: [12, 12, 5, 5]    — GPU 2,3 提前完成

Layer 2 dispatch (1 chunk):
  GPU 2 → GPU 3: 10 tokens

Greedy (LPT Layer 1):  GPU 2 avail=12, GPU 3 avail=12 → start=12, end=22
Oracle (Alt Layer 1):  GPU 2 avail=5,  GPU 3 avail=5  → start=5,  end=15

Greedy total: 22, Oracle total: 15
Oracle 改善: (22-15)/22 = 31.8%
```

**反例的核心逻辑**：LPT 将两个 size=5 的 chunk 排在 size=7 之后，导致 GPU 2 和 GPU 3 必须等待 GPU 0 和 GPU 1 完成大 chunk（因为 2→0 的 dst 是 GPU 0，必须等 GPU 0 完成 0→1 传输）。Oracle 先发出两个不冲突的小 chunk（2→0 和 1→3 涉及完全不同的 GPU pair），让 GPU 2 和 GPU 3 更早空闲，为 Layer 2 创造更有利的起始条件。

反例需要在代码中硬编码并运行，输出 greedy vs optimal 的具体调度时序。

### 代码

`experiments/poc_line1/simulator/oracle_simulator.py`，约 300 行。

---

## 七、Phase 4: 汇总报告

### 输出

```
outputs/poc_line1/report/
├── gate1_decision.json     # Gate 1 各指标值 + pass/fail
├── gate2_decision.json     # Gate 2 改善百分比 + pass/fail
├── overall_decision.md     # 综合判断 + 后续行动建议
├── figures/
│   ├── hit_rate_heatmap.png
│   ├── hit_rate_histogram.png
│   ├── decay_curve.png
│   ├── batch_size_vs_improvement.png
│   ├── placement_comparison.png
│   └── counterexample_timeline.png
└── raw_results/
    ├── cross_layer_stats.json
    ├── traffic_matrix_stats.json
    └── oracle_results.json
```

### Go/No-Go 决策矩阵

| Gate 1 | Gate 2 | 决策 |
|---|---|---|
| 通过 | 通过 | 全力推进 D11 + D10 + D20，进入实机实验 |
| 通过 | 灰色 | D11 保留但降级为辅助叙事，主线为 D10 + D20 |
| 通过 | 不通过 | 先排查建模问题（调度粒度、约束条件、placement 策略），修正后重新评估。修正后仍 < 5% 才放弃 D11，全力做 D10 + D20 |
| 灰色 | 通过 | D11 保留但需补充更多模型的跨层验证 |
| 灰色 | 灰色/不通过 | 放弃 D11，全力做 D10 + D20 |
| 不通过 | * | 放弃 D11，全力做 D10 + D20 |

---

## 八、目录结构

```
RS/experiments/poc_line1/
├── __init__.py
├── collection/
│   ├── __init__.py
│   └── full_trace_collector.py       # Phase 0: 全序列 trace 采集
├── analysis/
│   ├── __init__.py
│   ├── cross_layer_analyzer.py       # Phase 1: 跨层相关性分析
│   └── traffic_matrix_builder.py     # Phase 2: 流量矩阵构建
├── simulator/
│   ├── __init__.py
│   ├── scheduling_model.py           # 调度问题定义（Chunk, Constraint, Schedule, evaluate_schedule()）
│   ├── greedy_scheduler.py           # Per-layer LPT Greedy
│   ├── oracle_scheduler.py           # ILP Oracle + SA fallback
│   └── counterexample.py             # 2-layer 4-GPU 反例
├── reporting/
│   ├── __init__.py
│   ├── gate_report.py                # Gate 1/2 判定 + 报告生成
│   └── visualizer.py                 # 可视化图表生成
├── run_all.py                        # 统一入口脚本
└── run_collect.py                    # Phase 0 单独入口（需要 GPU）
```

---

## 九、执行方式

### 环境要求

- Phase 0: 单 GPU (≥16GB VRAM)，torch + transformers，OLMoE 模型权重
- Phase 1-4: 无 GPU 要求，Python 3.10+，依赖 numpy, scipy, matplotlib, pulp (ILP)

### 运行命令

```bash
# Phase 0 (需要 GPU 环境)
python experiments/poc_line1/run_collect.py \
  --model-id allenai/OLMoE-1B-7B-0924-Instruct \
  --prompts-path archive/poc1-20260629-local-prompts-0924/data/theory_prompts_50.jsonl \
  --num-prompts 32 \
  --output-dir outputs/poc_line1/full_sequence_trace

# Phase 1-4 (离线，任意环境)
python experiments/poc_line1/run_all.py \
  --trace-dir outputs/poc_line1/full_sequence_trace \
  --output-dir outputs/poc_line1
```

### POC1 遗留数据利用

Phase 0 采集完成后，POC1 的 ablation_checkpoint.jsonl 和 router_trace.json **仍然保留**作为交叉验证数据源：
- ablation 数据有 `delta_nll` 字段（expert importance 信号），可用于后续 D17 方向分析
- router_trace 数据（3 layers, 8 windows, token_pos=7）可作为 Phase 1 的补充数据点

---

## 十、已知局限性

1. OLMoE-1B-7B-0924 是 1B 参数模型，路由行为可能与更大规模模型（DeepSeek-V3 671B）有差异
2. 离线模拟使用 per-GPU half-duplex 并发约束模型近似跨机 Ethernet 场景，未建模 NCCL 内部的多 buffer pipeline 和 adaptive routing
3. 单请求 trace 不包含多请求并发时的流量叠加效应
4. expert placement 使用人工策略（round-robin / skewed），非生产环境的实际放置
5. combine 时间建模为 dispatch 转置，实际 combine 的流量模式可能因 expert output 大小差异而不同
6. Phase 0 的 trace 采集是 prefill 阶段的 autoregressive forward pass（`use_cache=False`，所有 token 同时输入），不包含 decode 阶段的逐 token 路由行为。按 position 切分（前 N vs 后 N）仅为近似分析，因为 prefill 中所有 token 均看到完整 prompt context，与 decode 阶段逐 token 生成的上下文不同。严格的 prefill vs decode 对比需额外采集 autoregressive decode trace。**sensitivity analysis**：对比 prefill trace 中"前 N 个 token 的 hit rate"和"后 N 个 token 的 hit rate"，如果差异不大，可初步论证 prefill 结论可外推到 decode
7. 未建模 GPU 计算时间与通信时间的重叠（expert forward 和下一层 dispatch 可能部分 overlap）
