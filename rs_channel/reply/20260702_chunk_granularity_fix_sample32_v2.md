# POC-line1 chunk 粒度修复验证

本轮修复目标：
- `barrier_aware_birkhoff`
- `randomized_multistart_birkhoff`
- `lagrangian`
- `two_stage`

修复原则：
- Birkhoff 分解只用于给原始 `(src, dst)` chunk 排序
- 不再使用 `_rounds_to_phase_order()` 生成新的细粒度 chunks 参与调度
- 所有候选继续通过 `_schedule_phase_orders()` 在统一模型下评估

## 验证命令

```bash
cd /root/autodl-tmp/RouterSense/RS
PYTHONPATH=src python -m pytest tests/test_poc_line1.py -q

OMP_NUM_THREADS=1 PYTHONPATH=src python -u experiments/poc_line1/pairwise_scheduler.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_batch500_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_batch500_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_batch500_v2/gate_weights.pt \
  --placement round_robin \
  --sample-limit 32 \
  --output-dir artifacts/poc_line1/candidate_screen_sample32_v2
```

## 结果位置

- `RS/artifacts/poc_line1/candidate_screen_sample32_v2/summary.json`

## 核心结论

1. 修复成功：之前超过 oracle 的 4 个候选都回到了 `oracle_perfect` 下方。
   - `oracle_perfect`: `40.92%`
   - `barrier_aware_birkhoff`: `22.68%`
   - `randomized_multistart_birkhoff`: `23.57%`
   - `lagrangian`: `24.16%`
   - `two_stage`: `16.55%`

2. 当前可信候选：
   - `fast_pairwise(best-of)`: `31.72%`, `2.94ms`
   - `simulated_annealing`: `26.76%`, `3.05ms`
   - `grasp`: `26.40%`, `3.04ms`
   - `ibbr`: `25.67%`, `2.60ms`

3. 主线结论不变：
   - `birkhoff`: `20.46%`
   - `oracle_perfect`: `40.92%`
   - 跨三 phase 联合调度相对逐层独立仍有约 `20` 个百分点额外空间。

4. 需要继续筛的边界项：
   - `lns`: `25.29%`, `5.57ms`
   - `lagrangian`: `24.16%`, `7.01ms`
   - `barrier_aware_birkhoff`: `22.68%`, `6.07ms`
   - `critical_path_compression`: `22.41%`, `0.94ms`

5. 已确认淘汰：
   - `lookahead_lpt`
   - `phase_aware_greedy`
   - `cp_lpt`
   - `completion_balanced`
   - `randomized_multistart_birkhoff`
   - `tabu_search`
   - `iterated_greedy`（仅因延迟过高）
