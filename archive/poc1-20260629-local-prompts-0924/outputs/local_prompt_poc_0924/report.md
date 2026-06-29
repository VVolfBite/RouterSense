# RouteSense PoC1 Report

## Summary

- Pairwise ranking accuracy: 0.4786421499292786
- Pairwise comparisons: 3535

## Strategies

- full: mean=0.000000, median=0.000000, p95=0.000000
- random: mean=0.007788, median=0.002670, p95=0.087891
- raw_routing: mean=0.011797, median=0.000641, p95=0.085938
- router_probability_min: mean=0.011797, median=0.000641, p95=0.085938
- effective_gate_weight_min: mean=0.011797, median=0.000641, p95=0.085938
- router_logit_min: mean=0.011797, median=0.000641, p95=0.085938
- topk_rank_max: mean=0.011797, median=0.000641, p95=0.085938
- top1_top2_gap_min: mean=0.030779, median=0.004150, p95=0.253906
- routing_entropy_max: mean=0.030779, median=0.004150, p95=0.253906
- abs_router_logit_min: mean=0.015907, median=0.000183, p95=0.117188
- oracle: mean=-0.078440, median=-0.031738, p95=-0.001129

## Routing Factor Diagnostics

- effective_gate_weight: pearson=-0.027566026119852968, spearman=-0.05538153093440299, pairwise=0.4786421499292786
- router_probability: pearson=-0.027566026119852968, spearman=-0.05538153093440299, pairwise=0.4786421499292786
- router_logit: pearson=0.009747517162991416, spearman=0.06959834066013577, pairwise=0.4786421499292786
- abs_router_logit: pearson=0.005113276401849055, spearman=0.029518991183784538, pairwise=0.5349363507779349
- topk_rank: pearson=-0.027303367150789775, spearman=-0.005618870926390765, pairwise=0.4786421499292786
- top1_top2_gap: pearson=-0.013698366279741709, spearman=-0.012138651314116133, pairwise=0.5123055162659123
- routing_entropy: pearson=0.06330175614116941, spearman=0.14530965777337693, pairwise=0.4876944837340877
- layer_id: pearson=-0.03439256281329183, spearman=-0.16431036129708498, pairwise=0.5123055162659123
- expert_id: pearson=0.020162068945170922, spearman=0.02134403384243433, pairwise=0.5123055162659123

## Conditional Diagnostics

- high_entropy pairwise=0.511641113003975, low_entropy pairwise=0.4458850056369786
- small_gap pairwise=0.5022650056625142, large_gap pairwise=0.4550593555681176
- large_magnitude pairwise=0.4132171387073348, small_magnitude pairwise=0.5203892493049119

## Oracle Subset

- top32 oracle subset pairwise=0.41037204058624577, oracle_mean_delta=-0.23235702514648438, random_mean_delta=-0.010039806365966797

## Notes

- 当前 PoC 每个 group 只取消一个 expert request，不直接代表最终通信节省比例。
- 当前取消 route 代表请求暂缓后最终不释放的质量上限近似。
- routing factor diagnostics 用于控制变量看单个 routing context 因子与 realized criticality 的关系。
