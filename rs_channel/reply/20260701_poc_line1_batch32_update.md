# POC-line1 Batch32 Update

本轮已按远端修正要求完成 P0/P1/P2 的关键执行，并产出新的真实结果。

## 已完成

1. 去除了旧的 fake oracle heuristic
   - 删除了 `oracle_schedule_multi_layer()` 中基于 `min(0.35, overlap_bonus)` 的硬编码改进公式。
   - 新实现优先使用单层 exact MILP。
   - 对 16 层整批问题，增加显式 `annealed_fallback`，不再伪造固定百分比收益。

2. 确认 OLMoE-1B-7B-0924-Instruct 的 router-active 层数
   - 单卡真实 probe 已跑。
   - 当前模型返回 `moe_layer_count = 16`。
   - `model.model.layers[0..15].mlp` 均为 `OlmoeSparseMoeBlock`，且都有 `gate` 和 `experts`。
   - 因此此前“只有 4 个 MoE 层”的假设不成立，必须按 16 层重新理解 Gate1 / Gate2。

3. 扩展为 32 prompt 批量 trace
   - 使用 `legacy/poc1/data/prompts/theory_prompts_50.jsonl` 的前 32 条 prompt。
   - 新入口支持 `--prompts-path` 和 `--num-prompts`。
   - 本次真实 trace 总记录数：`70272`

4. 修复 Gate1 rank correlation 的 placement 语义
   - 不再硬编码 `expert_id % 4`
   - 改为显式 `owner_by_expert`

5. 更新任务文档
   - `new-direction.md`
   - `poc-line-1.md`
   - 已写明当前 0924 checkpoint 是 16 个 router-active layers，而不是旧假设的 4 层。

## 本轮真实结果

### Trace / Probe

- trace_count: `32`
- moe_layer_count: `16`
- moe_layer_ids: `[0, 1, ..., 15]`
- topk: `8`
- record_count: `70272`

### Gate1: Cross-layer correlation

结果：

```json
{
  "passed": false,
  "pass_count": 0,
  "rank_pass": true
}
```

关键数值：

- `0->1` mean hit rate = `0.1232`
- `0->1` mean weighted hit rate = `0.0404`
- 最好的 pair 也只有：
  - `1->3` hit rate = `0.2033`
  - `8->12` hit rate = `0.1874`
- top rank correlation:
  - `5->6` spearman = `0.6236`
  - 其余大多远低于 0.3

结论：

- 16 层架构下，top-k expert overlap 非常低。
- 只有极少数局部层对在 rank-level flow 上显示中等正相关。
- 旧 Gate1 阈值在 16 层设定下明显过高，且“跨层 overlap 足够强”这一前提目前不成立。

### Gate2: Oracle vs Greedy

结果：

```json
{
  "mean_improvement_pct": 0.4838,
  "median_improvement_pct": 0.5309,
  "max_improvement_pct": 1.4644,
  "min_improvement_pct": -0.4215,
  "oracle_modes": ["annealed_fallback"],
  "gate2_decision": {
    "passed": false,
    "threshold_pct": 15.0
  }
}
```

结论：

- 之前 smoke 中出现的 `~35%` 提升完全来自 fake heuristic，不是真实 oracle 证据。
- 在 32 prompt、16 层真实 trace 上，当前 traffic model 下 oracle 相对 greedy 的改善接近 0。
- 这意味着当前 POC-line1 至少不能支持“强跨层联合调度空间”这一主张。

## 代码状态

本轮已完成并本地验证：

- `RS/src/routesense/evaluation/poc_line1.py`
- `RS/experiments/poc_line1/full_sequence_trace.py`
- `RS/experiments/poc_line1/cross_layer_analysis.py`
- `RS/tests/test_poc_line1.py`
- `new-direction.md`
- `poc-line-1.md`

测试：

- `cd RS && pytest -q` 通过
- `cd RS && pytest -q tests/test_poc_line1.py` 通过

## 结果路径

当前主结果：

- `RS/artifacts/poc_line1/full_sequence_trace_batch32/`
- `RS/artifacts/poc_line1/cross_layer_report_batch32_rr/`
- `RS/artifacts/poc_line1/oracle_report_batch32_rr/`

备份：

- `archive/backup/poc_line1/20260701_poc_line1_batch32/`

## 建议的下一步

当前更合理的方向不是继续强推“跨层预测 + 联合调度”。

更值得做的是：

1. 先把 Gate1/Gate2 重新定义为更弱、更局部的问题：
   - 只看相邻层局部 rank-flow correlation
   - 不再把跨多层 overlap 作为主证据

2. 或者转向新的主线：
   - 不再把 routing overlap 当核心信号
   - 改做更直接的 traffic/state shaping 或实时 flow-aware scheduling

3. 若仍保留 POC-line1，可继续：
   - 跑 `skewed` placement
   - 分 prompt / 分 token bucket / 分 layer subgroup 做条件分析
   - 看局部子集里是否有稳定信号

本轮按要求未 push。等待人工确认后再决定是否本地 commit / push。
