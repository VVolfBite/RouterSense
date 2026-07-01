# POC-LINE-1 Pairwise Oracle Rewrite Status

## 本轮完成

已按最新指令把 Gate2 的 pairwise oracle 从 phase-barrier 版本改为真正的 joint MILP。

### 代码变更

1. `RS/src/routesense/evaluation/poc_line1.py`
   - `greedy_schedule_pairwise(...)` 改为无全局 barrier 的连续推进版本
   - `pairwise_oracle(...)` 重写为 3-phase joint MILP：
     - phase0 = dispatch_L
     - phase1 = combine_L
     - phase2 = dispatch_{L+1}
   - 约束包括：
     - makespan bound
     - per-GPU half-duplex conflict (Big-M + binary)
     - per-GPU phase order: phase0 -> phase1 -> phase2
   - 新增 `evaluate_gate2(...)`
   - `run_pairwise_analysis(...)` 的 Gate2 summary 改为输出 decision / reason / thresholds

2. `RS/src/routesense/evaluation/__init__.py`
   - 导出 `evaluate_gate2`

3. `RS/tests/test_poc_line1.py`
   - 新增 / 修正 joint MILP 相关测试
   - 增加 “无全局 phase barrier” case
   - 增加 `evaluate_gate2` decision label 测试

## 当前测试

已运行：

```bash
cd /root/autodl-tmp/RouterSense/RS
pytest -q tests/test_poc_line1.py
```

结果：

```text
16 passed
```

## smoke 运行状态

已实际启动过：

```bash
python experiments/poc_line1/pairwise_scheduler.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_smoke_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_smoke_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_smoke_v2/gate_weights.pt \
  --placement round_robin \
  --output-dir artifacts/poc_line1/pairwise_oracle_report_smoke_v3
```

观察结果：

- 进程可正常启动，没有语义错误或立即异常退出
- `smoke_v3` 只有 `1 sample × 15 adjacent layer-pairs`
- 但 joint MILP 版本在当前实现下运行约 `3m49s` 仍未写出 `summary.json`
- 因此本轮已**主动停止** smoke 任务，避免继续空耗

当前结论：

- `pairwise_oracle` 的 barrier bug 已修正
- joint MILP 约束语义已通过单测
- 下一瓶颈是 solver 成本，不是 trace / routing / matrix 构造错误

## 本轮性能修复更新

已按最新指令进一步做了 4 项性能修复：

1. **同 phase 才生成 binary conflict pair**
   - 不再为跨 phase chunk 对生成 binary 变量

2. **phase ordering 改为辅助变量**
   - 用 `done[g][phase]` 取代 phase 间 chunk 叉积约束

3. **Big-M 收紧**
   - 从全 chunk 总和改为 per-GPU busy upper bound

4. **MILP 超时保护**
   - `options={"time_limit": 30}`
   - 超时不抛异常，返回 `makespan=None` 供上层 fallback

### 当前测试

```bash
cd /root/autodl-tmp/RouterSense/RS
pytest -q tests/test_poc_line1.py
```

结果：

```text
16 passed
```

### 性能修复后的 smoke 再运行

再次运行：

```bash
OMP_NUM_THREADS=1 python experiments/poc_line1/pairwise_scheduler.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_smoke_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_smoke_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_smoke_v2/gate_weights.pt \
  --placement round_robin \
  --output-dir artifacts/poc_line1/pairwise_oracle_report_smoke_v3
```

观察结果：

- 进程可正常启动
- 无语义错误、无立即异常退出
- 但 `smoke_v3`（仅 `1 sample × 15 adjacent layer-pairs`）在约 `1m53s` 时仍未写出 `summary.json`

### 当前判断

- 变量/约束爆炸问题已明显缓解
- 但当前主要瓶颈已变成 **SciPy/HiGHS 对 repeated pairwise MILP 的总体求解成本**
- 也就是说，问题已从“约束写错”推进为“求解器策略不够实用”

## 当前状态

- Gate1: 维持已完成状态，不改
- Gate2:
  - 旧的 barrier-based pairwise 语义已移除
  - 新的 joint MILP 已落地
  - 性能修复版 joint MILP 已落地
  - smoke 已二次验证，但当前 solver 成本仍偏高
  - 还没有拿到新的 smoke / batch500 完整结果

## 下一步（等开卡后执行）

优先顺序：

1. 继续优化 pairwise MILP 的执行策略或 fallback，再重跑 smoke：

```bash
python experiments/poc_line1/pairwise_scheduler.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_smoke_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_smoke_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_smoke_v2/gate_weights.pt \
  --placement round_robin \
  --output-dir artifacts/poc_line1/pairwise_oracle_report_smoke_v3
```

2. smoke 通过后再跑 batch500:

```bash
python experiments/poc_line1/pairwise_scheduler.py \
  --trace-jsonl artifacts/poc_line1/full_sequence_trace_batch500_v2/trace.jsonl \
  --hidden-states-path artifacts/poc_line1/full_sequence_trace_batch500_v2/hidden_states.pt \
  --gate-weights-path artifacts/poc_line1/full_sequence_trace_batch500_v2/gate_weights.pt \
  --placement round_robin \
  --output-dir artifacts/poc_line1/pairwise_oracle_report_batch500_v3
```

3. 如 round-robin 改善仍弱，再跑 `--placement skewed`

## 未执行

- 未得到新的 smoke 完整 summary
- 未跑 batch500 v3 完整结果
