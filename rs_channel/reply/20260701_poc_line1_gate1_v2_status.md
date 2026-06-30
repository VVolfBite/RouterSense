# POC-line1 Gate1 v2 Status

已按新 instruction 完成 P0 的核心修正，并做了真实单卡验证。

## 已完成

### 1. Phase 0 trace 扩展为 hidden-state observational trace

`RS/experiments/poc_line1/full_sequence_trace.py` 现在除了 `trace.jsonl` 外，还会输出：

- `hidden_states.pt`
- `gate_weights.pt`
- `architecture_probe.json`

其中：

- `hidden_states.pt` 保存每个 sample、每个 router-active layer 的 gate 输入 hidden states
- `gate_weights.pt` 保存每个 sample 对应层的 gate weight
- `architecture_probe.json` 明确记录每层 `mlp` 类型、是否有 `gate` / `experts`

### 2. 16 层架构确认

当前 `allenai/OLMoE-1B-7B-0924-Instruct` 的真实 probe 结果：

- `moe_layer_count = 16`
- `moe_layer_ids = [0, 1, ..., 15]`
- 所有层 `mlp_class = OlmoeSparseMoeBlock`
- 所有层 `has_gate = true`
- 所有层 `has_experts = true`
- `gate.weight.shape = [64, 2048]`

这说明当前 checkpoint 在 runtime 上表现为 16 个连续 router-active MoE layers，不是旧文档里假设的 4 层。

### 3. Gate1 语义已从“相似性”改为“可预测性”

旧 Gate1：

- 用 top-k overlap / hit rate 测两层 expert 选择相似度
- 该定义已被确认不对

新 Gate1：

- 用 `hidden_states[L_i]` 过 `gate_{L_{i+1}}`
- 得到 predicted top-k
- 与 `L_{i+1}` 实际 top-k 对比
- 主指标：`prefetch_accuracy = |predicted_topK ∩ actual_topK| / K`
- 同时记录 `cosine_similarity(hidden_i, hidden_{i+1})`

### 4. 单 prompt smoke（v2）

路径：

- `RS/artifacts/poc_line1/full_sequence_trace_smoke_v2/`
- `RS/artifacts/poc_line1/cross_layer_report_smoke_v2/`

结果：

```json
{
  "passed": true,
  "pass_count": 12,
  "rank_pass": true
}
```

代表性 layer pair：

- `8->9`: prefetch_accuracy `0.9013`, cosine `0.8962`
- `13->14`: prefetch_accuracy `0.9013`, cosine `0.9087`
- `9->10`: prefetch_accuracy `0.8882`, cosine `0.8861`

### 5. 32 prompt batch（v2）

路径：

- `RS/artifacts/poc_line1/full_sequence_trace_batch32_v2/`
- `RS/artifacts/poc_line1/cross_layer_report_batch32_rr_v2/`

结果：

```json
{
  "passed": true,
  "pass_count": 13,
  "rank_pass": true
}
```

代表性 layer pair：

- `8->9`: prefetch_accuracy `0.8843`, cosine `0.9000`
- `9->10`: prefetch_accuracy `0.8773`, cosine `0.9033`
- `7->8`: prefetch_accuracy `0.8732`, cosine `0.8911`

较弱但仍可见的 pair：

- `1->2`: prefetch_accuracy `0.5649`, cosine `0.7421`
- `0->1`: prefetch_accuracy `0.4429`, cosine `0.6629`

## 关键结论

旧 Gate1 的“失败”主要是指标定义错了。

- 如果用 overlap/hit-rate 测相似性，Gate1 在 batch32 上失败
- 如果改用 Fate 风格的 prefetch accuracy 测可预测性，Gate1 在 smoke 和 batch32 上都通过

这意味着：

1. `两层选不同 expert` 不等于 `下一层不可预测`
2. 对当前 OLMoE-0924，跨层 gate prediction 至少在相邻层上是强信号
3. POC-line1 主线可以继续，但必须建立在新 Gate1 语义上

## 仍未完成

### Gate2 仍需继续修

当前 Oracle 路径已经移除了旧的 fake 35% heuristic。

现状：

- 单层 exact MILP 已接入
- 多层大实例目前走 `annealed_fallback`
- 旧 batch32 `oracle_vs_greedy` 的真实结果只有 `~0.48%` 改善

但远端新 instruction 对 Gate2 的要求是：

- 需要更严格的 Oracle / ILP / SA 说明
- 需要和新版 Gate1 / 16 层架构一起重新评估

因此当前阶段我认为：

- **P0 已完成且结论明确**
- **P1/P3/P4/P5/P6 只完成了部分铺垫，尚未全部关掉**

## 本轮新增结果路径

主结果：

- `RS/artifacts/poc_line1/full_sequence_trace_smoke_v2/`
- `RS/artifacts/poc_line1/cross_layer_report_smoke_v2/`
- `RS/artifacts/poc_line1/full_sequence_trace_batch32_v2/`
- `RS/artifacts/poc_line1/cross_layer_report_batch32_rr_v2/`

备份：

- `archive/backup/poc_line1/20260701_poc_line1_batch32_v2/`

## 建议的下一步

1. 用新版 Gate1 结果更新 task 文档和 go/no-go 叙事
2. 继续收紧 Gate2：
   - 明确 exact / fallback 边界
   - 批量 oracle 重新评估
3. 再决定是否扩到 500 prompt exploration

本轮未 push，等待后续指令。
