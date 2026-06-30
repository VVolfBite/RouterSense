# POC-LINE-1 Pairwise Oracle Update

## 本轮完成

已补齐离线 Gate2 主链的缺口，不依赖 GPU：

1. 在 `RS/src/routesense/evaluation/poc_line1.py` 中新增：
   - `build_predicted_traffic(...)`
   - `greedy_schedule_pairwise(...)`
   - `pairwise_oracle(...)`
   - `run_pairwise_analysis(...)`

2. 新增实验入口：
   - `RS/experiments/poc_line1/pairwise_scheduler.py`

3. 测试补齐：
   - predicted traffic matrix 构建
   - pairwise oracle 相对 greedy 的单调性
   - pairwise summary / gate2_decision 生成

4. 当前 `cd RS && pytest -q` 通过：
   - `33 passed`

## 当前语义

新的 Gate2 离线链现在是：

1. `hidden_i -> gate_{i+1}` 生成 predicted top-k
2. 从 predicted top-k 构建 `T_{i+1}^{pred}`
3. 对每个相邻层对 `(L_i, L_{i+1})` 独立比较：
   - `greedy_schedule_pairwise(T_i, C_i, T_{i+1}^{actual})`
   - `pairwise_oracle(T_i, C_i, T_{i+1}^{actual})`
   - `pairwise_oracle(T_i, C_i, T_{i+1}^{pred})`
4. 输出：
   - `perfect_improvement_pct`
   - `predicted_improvement_pct`
   - `traffic_correlation`
   - `predicted_improvement_vs_traffic_correlation`

## 仍未做

还没有重新跑真实 batch32 / batch500 的 pairwise oracle 结果。
也还没把新的 pairwise 结果写回 `archive/backup`。

## 建议下一步

优先用现有真实产物直接运行：

```bash
cd /root/autodl-tmp/RouterSense/RS
python experiments/poc_line1/pairwise_scheduler.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_batch500_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_batch500_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_batch500_v2/gate_weights.pt \
  --placement round_robin \
  --output-dir artifacts/poc_line1/pairwise_oracle_report_batch500_rr_v2
```

如需进一步放大 Gate2，可再跑 `--placement skewed`。
