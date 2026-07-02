# POC-line1 候选调度算法 32/64 sample screening

本轮使用：
- `RS/artifacts/poc_line1/candidate_screen_sample32_v1/summary.json`
- `RS/artifacts/poc_line1/candidate_screen_sample64_v1/summary.json`

主判据采用 `sample64`，`sample32` 用于稳定性复核。

## 核心结果

1. 主线结论继续成立。
   - `birkhoff` mean improvement: `19.99%`
   - `oracle_perfect` mean improvement: `40.17%`
   - 跨三 phase 联合调度相对 FAST 式逐层独立，仍有约 `20` 个百分点额外空间。

2. 预测信号仍然有效。
   - `oracle_predicted`: `40.39%`
   - `oracle_perfect`: `40.17%`
   - `sample64` 中 solver status:
     - perfect: `{"OPTIMAL": 960}`
     - predicted: `{"OPTIMAL": 960}`
   - `sample32` 中 predicted 有 `1` 个 `UNKNOWN`，但不影响总体结论。

3. 当前可信的实用候选是：
   - `simulated_annealing`: `26.33%`, `3.04ms`
   - `grasp`: `25.67%`, `3.05ms`
   - `ibbr`: `25.09%`, `2.53ms`

4. 一批新候选结果越过了 oracle 上界，当前不能视为有效：
   - `lagrangian`: `68.40%`, `5.23ms`
   - `two_stage`: `54.12%`, `0.24ms`
   - `fast_pairwise(best-of)`: `63.27%`, `2.36ms`
   这说明实现/评估口径还不一致，不能直接进入论文主表。

5. 明确淘汰项：
   - `lookahead_lpt`
   - `phase_aware_greedy`
   - `cp_lpt`
   - `completion_balanced`
   - `barrier_aware_birkhoff`
   - `randomized_multistart_birkhoff`

## 归档位置

- 淘汰表：`/root/autodl-tmp/RouterSense/ELIMINATION_LOG.md`
- sample32 结果：`/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/candidate_screen_sample32_v1/`
- sample64 结果：`/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/candidate_screen_sample64_v1/`

## 建议下一步

1. 暂时把主表只保留：
   - `greedy`
   - `birkhoff`
   - `simulated_annealing`
   - `grasp`
   - `ibbr`
   - `oracle_perfect`
   - `oracle_predicted`

2. 单独修正：
   - `lagrangian`
   - `two_stage`
   - `fast_pairwise` 组合器

3. 后续算法筛选默认先跑 `sample_limit=32` 或 `64`，不再直接上 `500`。
