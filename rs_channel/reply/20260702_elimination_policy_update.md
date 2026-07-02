# POC-line1 淘汰策略修正

已根据最新指示更新 `ELIMINATION_LOG.md`，核心变化：

1. 不再在 `sample32` 阶段过早淘汰第二/第三梯队算法。
2. 当前只明确淘汰：
   - `lookahead_lpt`
   - `phase_aware_greedy`
   - `cp_lpt`
   - `completion_balanced`
3. 以下算法统一改为“待定”，保留至大规模验证：
   - `tabu_search`
   - `iterated_greedy`
   - `randomized_multistart_birkhoff`
   - `lns`
   - `barrier_aware_birkhoff`
   - `lagrangian`
   - `critical_path_compression`
   - `two_stage`
   - `cp_local_swap`

## 新判定规则

- 保留：改善 ≥25% 且延迟 ≤5ms
- 待定：改善 ≥15% 或延迟 ≤10ms
- 淘汰：改善 <10% 或稳定劣于 greedy

## 文件位置

- `/root/autodl-tmp/RouterSense/ELIMINATION_LOG.md`

## 后续建议

1. 先用 `sample_limit=32/64` 继续小样本筛算法。
2. 再做 `N=8/16 GPU` 或更大规模复杂度分析。
3. 等扩展性验证后，再决定是否真正淘汰高延迟但高改善的候选。
