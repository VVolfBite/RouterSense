# 调度算法淘汰清单

基于：
- `RS/artifacts/poc_line1/candidate_screen_sample32_v2/summary.json`

本表已按 chunk 粒度修复后的 `sample32_v2` 更新。此前 `v1` 中超过 oracle 的候选，已确认为 chunk 拆分导致的无效结果。

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
| randomized_multistart_birkhoff | 23.57% | 20.11ms | 淘汰 | 修复后收益正常，但延迟太高 |
| tabu_search | 28.33% | 20.59ms | 淘汰 | 收益可以，但被更快候选支配 |
| lns | 25.29% | 5.57ms | 待定 | 收益过线，但延迟略高于 5ms |
| simulated_annealing | 26.76% | 3.05ms | 保留 | 目前最稳妥的新候选之一 |
| lagrangian | 24.16% | 7.01ms | 待定 | 修复后已回到 oracle 下方，但暂未达到主目标 |
| grasp | 26.40% | 3.04ms | 保留 | 与 simulated_annealing 接近，性价比好 |
| completion_balanced | 1.72% | 0.17ms | 淘汰 | 收益太低 |
| two_stage | 16.55% | 0.34ms | 待定 | 修复后不再异常，但提升一般 |
| critical_path_compression | 22.41% | 0.94ms | 待定 | 比 birkhoff 略好，但未达到 25% |
| ibbr | 25.67% | 2.60ms | 保留 | 过线且延迟可接受 |
| iterated_greedy | 31.56% | 24.58ms | 淘汰（离线候选） | 收益强，但延迟远超 5ms，只适合作为离线参考 |
| cp_local_swap | 16.21% | 3.27ms | 待定 | 有正收益，但提升一般 |
| fast_pairwise(best-of) | 31.72% | 2.94ms | 保留（组合器） | 修复后 best-of 回到合理区间，当前主力组合器 |

## 当前结论

- 主线结论成立：`oracle_perfect` 40.92% vs `birkhoff` 20.46%，跨层联合调度相对逐层独立仍有约 20 个百分点的额外空间。
- 目前可信的实用候选集中在：
  - `fast_pairwise(best-of)`
  - `simulated_annealing`
  - `grasp`
  - `ibbr`
- `barrier_aware_birkhoff`、`randomized_multistart_birkhoff`、`lagrangian`、`two_stage` 的 chunk 粒度问题已修复，不再超过 oracle 上界。

## 判定规则

- **保留**：改善 ≥25% 且延迟 ≤5ms
- **淘汰**：改善 <15% 或延迟 >10ms 或被其他算法支配
- **待定**：介于两者之间，需进一步调参
