# 调度算法候选清单

本文件不再作为“最终淘汰清单”，而作为当前阶段的**三档候选池**。

原因：
- 现在已经确认 `atomic/chunk` 与 `wave/fluid` 是独立评价维度；
- 直接用旧的单一“淘汰/保留”框架会混淆方法收益与调度语义收益；
- 当前更合理的做法，是按**研究阶段目标**维护候选池。

## 统一语义

- `B_*`：外部/基线方法线，尤其是逐 phase 本地优化的强基线
- `U_*`：我们的方法线，只要调度决策会显式利用其他 phase 信息，就归入 `U`
- `*_atomic`：chunk/atomic 语义
- `*_wave`：wave/fluid 语义

## Tier 1：定论候选池

用途：
- 只从这里选 `3` 个左右代表方法，做最终定论级对比
- 目标是覆盖不同维度，而不是机械选分数前三

实验规格：
- `sample >= 256`
- `N >= 16`
- 优先看 `N=16 / 32`

推荐保留：

| 算法 | 角色 | 当前原因 |
|------|------|----------|
| `B_birkhoff` | `B_atomic` 强基线 | 最干净的逐 phase atomic 基线 |
| `B_birkhoff_wave` | `B_wave` 强基线 | 用来隔离 wave 语义本身带来的收益 |
| `U_gated_maxweight_matching` | `U_wave` 主候选 | 当前最标准的全局 ready-set max-weight 主方法 |
| `U_barrier_criticality_global_matching` | `U_wave` 主候选 | 更强调 barrier unlock 的联合调度候选 |
| `U_gated_maxweight_matching_atomic` | `U_atomic` 主候选 | 用来隔离“联合决策”本身而非 fluid 分流 |
| `U_barrier_criticality_global_matching_atomic` | `U_atomic` 主候选 | 同上，保留第二个联合 atomic 代表 |
| `U_lagrangian` | 旧一代联合方法 | 作为非 matching-based 的 `U` 代表 |

建议最终打架时优先从这里挑三类：
- `B_atomic`：`B_birkhoff`
- `B_wave`：`B_birkhoff_wave`
- `U_wave`：`U_gated_maxweight_matching` 或 `U_barrier_criticality_global_matching`

必要时额外补：
- `U_atomic`：`U_gated_maxweight_matching_atomic`

## Tier 2：优化晋级池

用途：
- 看有没有值得继续打磨、晋级到 Tier 1 的算法
- 更关注机制潜力，不直接下最终结论

实验规格：
- `N=8`
- `sample=64`

当前候选：

| 算法 | 当前定位 |
|------|----------|
| `U_cp_lpt` | 最早一代利用三层信息排序的联合方法 |
| `U_lagrangian` | 旧联合方法，可能仍有进一步调参空间 |
| `U_ibbr` | repair 型联合启发式 |
| `U_gated_greedy_maximal` | 低成本 ready-set baseline |
| `U_gated_greedy_maximal_atomic` | 上述方法的 atomic 对照 |
| `U_barrier_price_adaptive_matching` | barrier price 自适应版本 |
| `U_barrier_price_adaptive_matching_atomic` | 上述方法的 atomic 对照 |
| `B_barrier_aware_birkhoff` | 比 `B_birkhoff` 更强的 phase-local 基线 |
| `B_barrier_aware_birkhoff_wave` | 上述基线的 wave 版本 |
| `pairwise_wave_oracle` | atomic wave 强参考，不作为部署候选 |

## Tier 3：快速淘汰池

用途：
- 快速试错，不追求结论稳定性
- 主要目的是尽快淘汰第一次尝试但明显不合适的算法

实验规格：
- `N=4`
- `sample=32`

适合放在这里的：

| 算法 | 当前定位 |
|------|----------|
| `lookahead_lpt` | 旧快速启发式，历史上常劣于更强基线 |
| `completion_balanced` | 收益偏低，主要保留作机制参考 |
| `critical_path_compression` | 收益有限，更多是局部调序尝试 |
| `cp_local_swap` | 小修补方法，常被更强 repair 法支配 |
| `quantized_decomposed` | 当前已明显不合适，仅保留代码参考 |
| `tabu_search` | 若继续尝试新邻域，可先回 Tier 3 快速筛 |
| `lns` | 同上 |
| `simulated_annealing` | 同上 |
| `grasp` | 同上 |
| `decomposed` | 需在明确 merge 语义后再考虑回升 |

## Oracle / Reference

这些不是部署候选，但需要保留作为参考：

| 算法 | 作用 |
|------|------|
| `pairwise_oracle` | 现有 `atomic/chunk` CP-SAT 上界 |
| `pairwise_wave_oracle` | 可进入中等规模实验的 atomic wave 强参考 |
| `pairwise_fluid_wave_oracle` | 仅用于小样本上限分析的 fluid/wave 上界原型 |

注意：
- `pairwise_oracle` 仍是 `atomic/chunk` 语义；
- `pairwise_fluid_wave_oracle` 不能直接拿去作为大规模部署结论，只能做小样本理论上限说明。

## 当前阶段结论

1. 现在不适合直接下“大规模最终定论”。
2. 首先需要把收益拆成三部分：
   - `B_atomic -> B_wave`
   - `B_wave -> U_atomic`
   - `U_atomic -> U_wave`
3. 当前最值得继续投资源的是：
   - `B_birkhoff_wave`
   - `U_gated_maxweight_matching`
   - `U_barrier_criticality_global_matching`
   - 以及对应的 atomic 对照

## 当前唯一标准语料

当前仅保留一个标准非重复 prompt 集：

- `RS/artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl`

用途：
- Tier 1 定论级实验的统一语料入口
- 后续 `sample=64` / `sample=32` 仅从该 256 集合取前缀子集，不再维护多份独立 prompt 集
