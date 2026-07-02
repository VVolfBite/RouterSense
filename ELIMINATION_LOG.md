# 调度算法淘汰清单

基于：
- `RS/artifacts/poc_line1/candidate_screen_sample32_v1/summary.json`
- `RS/artifacts/poc_line1/candidate_screen_sample64_v1/summary.json`

下表使用 `sample64` 作为主判据，`sample32` 作为稳定性复核。

| 算法 | 改善(vs greedy) | 延迟 | 状态 | 淘汰原因/保留理由 |
|------|----------------|------|------|-----------------|
| greedy | 0.00% | 0.10ms | 基线 | 无优化基线 |
| birkhoff | 19.99% | 0.56ms | 保留（FAST对标） | 逐层独立强基线，后续所有跨层算法都要对它比较 |
| oracle_perfect | 40.17% | 25.71ms | 保留（上界） | 跨三 phase 联合最优上界 |
| oracle_predicted | 40.39% | 28.33ms | 保留（预测上界） | 与 perfect 几乎重合，说明预测信号有效 |
| lookahead_lpt | -11.28% | 0.22ms | 淘汰 | 稳定劣于 greedy |
| cp_lpt | 5.95% | 0.25ms | 淘汰 | 收益太低 |
| phase_aware_greedy | -9.19% | 0.24ms | 淘汰 | 稳定劣于 greedy |
| barrier_aware_birkhoff | -2717.86% | 3.51ms | 淘汰 | 明显实现错误，结果远差于基线 |
| randomized_multistart_birkhoff | -1251.71% | 10.16ms | 淘汰 | 明显实现错误，且延迟过高 |
| tabu_search | 27.67% | 19.87ms | 淘汰 | 收益可以，但被 simulated_annealing / grasp 支配 |
| lns | 24.73% | 5.55ms | 待定 | 接近门槛，但略低于 25% 且略高于 5ms |
| simulated_annealing | 26.33% | 3.04ms | 保留 | 目前最稳妥的新候选之一 |
| lagrangian | 68.40% | 5.23ms | 淘汰（实现失真） | 超过 oracle 上界，说明当前实现/评估不一致，不能当有效结果 |
| grasp | 25.67% | 3.05ms | 保留 | 与 simulated_annealing 接近，性价比好 |
| completion_balanced | 0.81% | 0.17ms | 淘汰 | 收益太低 |
| two_stage | 54.12% | 0.24ms | 淘汰（实现失真） | 超过 oracle 上界，当前实现无效 |
| critical_path_compression | 21.55% | 0.93ms | 待定 | 比 birkhoff 略好，但未达到 25% |
| ibbr | 25.09% | 2.53ms | 保留 | 刚过阈值，且 32/64 两轮稳定 |
| iterated_greedy | 30.73% | 23.72ms | 淘汰（离线候选） | 收益强，但延迟远超 5ms，只适合作为离线参考 |
| cp_local_swap | 15.48% | 3.22ms | 待定 | 有正收益，但提升一般 |
| fast_pairwise(best-of) | 63.27% | 2.36ms | 淘汰（组合失真） | 组合器被无效候选污染，当前 best-of 结果不可直接使用 |

## 当前结论

- 主线结论成立：`oracle_perfect` 40.17% vs `birkhoff` 19.99%，跨层联合调度相对逐层独立仍有约 20 个百分点的额外空间。
- 目前可信的实用候选集中在：
  - `simulated_annealing`
  - `grasp`
  - `ibbr`
- 目前不可信的高收益候选：
  - `lagrangian`
  - `two_stage`
  - `fast_pairwise(best-of)`
  这些结果超过了 oracle 上界，必须视为实现或评估不一致，而不是“更强算法”。

## 判定规则

- **保留**：改善 ≥25% 且延迟 ≤5ms
- **淘汰**：改善 <15% 或延迟 >10ms 或被其他算法支配
- **待定**：介于两者之间，需进一步调参
