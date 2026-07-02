# 调度算法淘汰清单

基于：
- `RS/artifacts/poc_line1/candidate_screen_sample32_v2/summary.json`

本表已按 chunk 粒度修复后的 `sample32_v2` 更新。此前 `v1` 中超过 oracle 的候选，已确认为 chunk 拆分导致的无效结果。

## 重要说明

当前评估基于 `sample32`（`N=4 GPU`），尚未在大规模场景下验证。
所有“待定”算法保留至：
1. 完成 `N=8/16 GPU` 的大规模测试
2. 完成各算法的时间复杂度分析（`O(N)` 对比）
3. 确认算法在规模扩展下的延迟增长趋势

| 算法 | 改善(vs greedy) | 延迟 | 状态 | 淘汰原因/保留理由 |
|------|----------------|------|------|-----------------|
| greedy | 0.00% | 0.12ms | 基线 | 无优化基线 |
| birkhoff | 20.46% | 0.67ms | 保留（FAST对标） | 逐层独立强基线，后续所有跨层算法都要对它比较 |
| oracle_perfect | 40.92% | 29.68ms | 保留（上界） | 跨三 phase 联合最优上界 |
| oracle_predicted | 41.12% | 33.04ms | 保留（预测上界） | 与 perfect 几乎重合，说明预测信号有效 |
| lookahead_lpt | -10.11% | 0.26ms | 淘汰 | 稳定劣于 greedy |
| cp_lpt | 6.74% | 0.29ms | 淘汰 | 收益太低 |
| phase_aware_greedy | -8.06% | 0.28ms | 淘汰 | 稳定劣于 greedy |
| barrier_aware_birkhoff | 22.68% | 6.07ms | 待定 | 已修复为合理值，但收益未过 25%，延迟略高 |
| randomized_multistart_birkhoff | 23.57% | 20.11ms | 待定 | 与 barrier_aware_birkhoff 同类，延迟问题待规模验证 |
| tabu_search | 28.33% | 20.59ms | 待定 | 改善率高，当前只在小规模测过，延迟扩展性未验证 |
| lns | 25.29% | 5.57ms | 待定 | 收益过线，但延迟略高于 5ms |
| simulated_annealing | 26.76% | 3.05ms | 保留 | 目前最稳妥的新候选之一 |
| lagrangian | 24.16% | 7.01ms | 待定 | 修复后已回到 oracle 下方，但暂未达到主目标 |
| grasp | 26.40% | 3.04ms | 保留 | 与 simulated_annealing 接近，性价比好 |
| completion_balanced | 1.72% | 0.17ms | 淘汰 | 收益太低 |
| two_stage | 16.55% | 0.34ms | 待定 | 延迟极低，仍有调参和大规模验证价值 |
| critical_path_compression | 22.41% | 0.94ms | 待定 | 延迟极低，改善率仍有潜力 |
| ibbr | 25.67% | 2.60ms | 保留 | 过线且延迟可接受 |
| iterated_greedy | 31.56% | 24.58ms | 待定 | 收益最高的非 oracle 候选之一，需等规模验证后再定性 |
| cp_local_swap | 16.21% | 3.27ms | 待定 | 延迟合理，改善率一般，但仍有潜力 |
| fast_pairwise(best-of) | 31.72% | 2.94ms | 保留（组合器） | 修复后 best-of 回到合理区间，当前主力组合器 |

## 当前结论

- 主线结论成立：`oracle_perfect` 40.92% vs `birkhoff` 20.46%，跨层联合调度相对逐层独立仍有约 20 个百分点的额外空间。
- 目前可信的主候选集中在：
  - `fast_pairwise(best-of)`
  - `simulated_annealing`
  - `grasp`
  - `ibbr`
- 第二/第三梯队当前不淘汰，继续保留做规模验证：
  - `tabu_search`
  - `iterated_greedy`
  - `randomized_multistart_birkhoff`
  - `lns`
  - `barrier_aware_birkhoff`
  - `lagrangian`
  - `critical_path_compression`
  - `two_stage`
  - `cp_local_swap`
- `barrier_aware_birkhoff`、`randomized_multistart_birkhoff`、`lagrangian`、`two_stage` 的 chunk 粒度问题已修复，不再超过 oracle 上界。

## 判定规则

- **保留**：改善 ≥25% 且延迟 ≤5ms
- **待定**：改善 ≥15% 或延迟 ≤10ms
- **淘汰**：改善 <10% 或稳定劣于 greedy
