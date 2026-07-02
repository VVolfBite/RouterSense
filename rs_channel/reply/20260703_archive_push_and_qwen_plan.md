本轮归档整理已推送完成。

- commit: `3be3c39`
- branch: `main`
- remote: `origin`

关键点：

1. `archive/backup/` 现在只保留 3 个核心实验存档：
   - `20260703_cross_layer_prediction_validity`
   - `20260703_oracle_fast_gap_study`
   - `20260703_execution_window_multiscale`

2. 顶层历史任务文档已归档到：
   - `archive/backup/docs/`

3. 为避免 push 过慢，归档中不再保留超大 raw payload：
   - cross-layer 的 `prediction_rows.json`
   - oracle/fast gap 的两个超大 `results.json`

4. 当前保留的是轻量且可审查的内容：
   - `README.md`
   - `docs/*.md`
   - `reports/*.md`
   - 小型 `summary.json` / `table.md` / `table_e2e.md`
   - 必要的 `strategy_summary.json` / `placement.json` / `perf.log`

Qwen 验证规划：

1. 不重写评估框架。
2. 保持 `run_pairwise_analysis()`、candidate compare、scheduler 逻辑不变。
3. 只补一个 Qwen-MoE trace collector，输出与 OLMoE 完全一致的产物：
   - `trace.jsonl`
   - `hidden_states.pt`
   - `gate_weights.pt`
   - `summary.json`
4. 先做 architecture probe，确认：
   - router-active layers 数量
   - `topk`
   - gate module 路径
   - router logits 返回方式
5. 再用同一套 execution-window 候选在 Qwen trace 上复跑，重点看：
   - `B_birkhoff`
   - `O_lagrangian`
   - `O_ibbr`
   - `O_gated_maxweight_matching`
   - `O_barrier_criticality_global_matching`
   - `O_barrier_price_adaptive_matching`

目标是验证：主方法相对强基线的收益趋势是否跨模型仍成立。
