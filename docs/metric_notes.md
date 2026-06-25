# RouteSense POC1 指标说明

这份说明解释当前 POC1 本机阶段最常看的指标。它的目标不是给论文级定义，而是让后续回看结果时能快速判断每个数字大概说明什么。

## Trace / Batch 基础指标

`active_destination_count` 表示当前 batch 中实际出现 token 路由的 destination 数量。它越大，说明这批 token 分散到更多目标上，通信和计算的结构更复杂。

`hot_bucket_count` 表示 token 数明显偏高的 bucket 数量。它用来粗略判断是否存在多个热点，而不是只有一个最大桶。

`max_bucket_token_count` 表示当前样本里最大 bucket 的 token 数。它是最直观的负载峰值 proxy。

`avg_bucket_token_count` 表示非空 bucket 的平均 token 数。它用于和最大 bucket 对比，判断负载是不是集中。

`max_to_avg_bucket_ratio` 是最大 bucket token 数除以平均 bucket token 数。它越高，说明最大桶相对普通桶更突出，但它只看大小，不直接说明依赖结构。

`bucket_token_count_gini` 是 bucket token 数的不均衡程度。越接近 0 越均匀，越高说明负载越集中。

`routing_entropy` 表示 router 概率分布的熵。它越高，通常说明 router 对候选 expert 的分布更分散；越低说明更集中。

`top1_top2_gap` 表示 router top-1 和 top-2 概率差。它越大，说明 router 对第一选择更有把握；越小则说明 top choices 更接近。

## Proxy Compare 指标

`state_dependency_top1_different_rate` 表示 state-only 与 dependency-only 排出的 top-1 bucket 有多少比例不同。它回答“二者是否经常给出不同第一选择”，但不回答谁更好。

`full_state_top1_different_rate` 表示 full proxy 与 state-only 的 top-1 不同率。它用于观察加入 dependency 信号后，最终组合分数是否改变了 state-only 的决策。

`mean_state_dependency_topk_jaccard` 表示 state-only 和 dependency-only 的 top-k bucket 集合平均 Jaccard 相似度。越接近 1，说明二者 top-k 集合越接近；越低说明排序候选差异越明显。

`mean_state_full_topk_jaccard` 表示 state-only 和 full proxy 的 top-k 集合平均 Jaccard 相似度。它用于观察 full proxy 是否只是复制 state-only，还是引入了新的候选偏好。

## Critical Bucket 指标

`state_top1_hit_rate` 表示 state-only 的 top-1 bucket 命中 pseudo critical top-1 bucket 的比例。它回答“只看状态/大小类信号时，能不能挑中疑似关键桶”。

`dependency_top1_hit_rate` 表示 dependency-only 的 top-1 命中比例。它用于判断依赖结构信号是否更接近疑似 barrier-causing bucket。

`full_top1_hit_rate` 表示 full proxy 的 top-1 命中比例。它通常是当前最重要的决策指标之一，因为 full proxy 代表状态和依赖信号的简单组合。

`dependency_mean_rank_of_critical_top1` 表示 pseudo critical top-1 bucket 在 dependency-only 排序中的平均名次。越接近 1 越好，说明 dependency-only 更早把疑似关键桶排到前面。

`full_mean_rank_of_critical_top1` 表示 pseudo critical top-1 bucket 在 full 排序中的平均名次。它越低，说明 full proxy 对关键桶的优先级越高。

`pseudo ground truth v1` 是偏结构型的疑似关键桶定义。它会更多使用 bridge、coverage、position spread 等 dependency 风格特征，因此适合看结构信号是否有效，但独立性较弱。

`pseudo ground truth v2` 是更偏 load-centric 的疑似关键桶定义。它提高 bucket size、selected route share、destination load、token coverage 等更中性的负载特征权重，并降低 bridge/position 类特征权重。它更适合做当前阶段的后续决策 proxy。

## Grouping / Layer 指标

`most_informative_layer` 表示当前 proxy 指标里差异或命中增量最明显的 MoE 层。它用于决定下一轮实验优先追哪一层。

`least_informative_layer` 表示当前信号最弱的层。它不代表该层没有任何意义，只表示当前本机 proxy 结果里不值得优先投入。

`strongest_grouping_dimension` 表示哪一种分组维度最能放大 proxy 差异或命中差异。例如 skew 分组不一定最强，`mean_bridge_score`、`top3_bucket_mass` 或 `active_destination_count` 可能更接近真正短板结构。

## 当前阅读建议

先看 `critical_bucket_proxy_summary.json` 里的 v2 overall hit rate，再看 per-layer hit rate，最后看 grouping 维度。当前阶段最重要的问题不是“排序是否分叉”，而是“哪种 proxy 更接近疑似关键桶”。
