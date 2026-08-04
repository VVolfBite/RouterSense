# RouteSense PoC1 Report

## Summary

- Pairwise ranking accuracy: 0.4633093525179856
- Pairwise comparisons: 3475

## Strategies

- full: mean=0.000000, median=0.000000, p95=0.000000
- random: mean=0.000758, median=-0.000019, p95=0.044434
- raw_routing: mean=-0.001098, median=-0.000002, p95=0.019531
- router_probability_min: mean=-0.001098, median=-0.000002, p95=0.019531
- effective_gate_weight_min: mean=-0.001098, median=-0.000002, p95=0.019531
- router_logit_min: mean=-0.001098, median=-0.000002, p95=0.019531
- topk_rank_max: mean=-0.001088, median=-0.000002, p95=0.019531
- top1_top2_gap_min: mean=-0.010736, median=-0.000239, p95=0.084473
- routing_entropy_max: mean=-0.010736, median=-0.000239, p95=0.084473
- abs_router_logit_min: mean=0.002152, median=-0.000000, p95=0.053711
- oracle: mean=-0.066552, median=-0.007202, p95=-0.000005

## Routing Factor Diagnostics

- effective_gate_weight: pearson=-0.0063444782923299005, spearman=-0.06934309474726713, pairwise=0.4633093525179856
- router_probability: pearson=-0.0063444782923299005, spearman=-0.06934309474726713, pairwise=0.4633093525179856
- router_logit: pearson=-0.0004870398257390361, spearman=0.07597386351235581, pairwise=0.4633093525179856
- abs_router_logit: pearson=-0.014201371678276942, spearman=-0.07218523965923779, pairwise=0.5162589928057554
- topk_rank: pearson=0.0354721982138192, spearman=0.05242787810654466, pairwise=0.46302158273381294
- top1_top2_gap: pearson=0.03478502978196987, spearman=-0.031571728103323905, pairwise=0.517410071942446
- routing_entropy: pearson=-0.0728767674710301, spearman=0.06382925289782354, pairwise=0.48258992805755396
- layer_id: pearson=0.03777238801706386, spearman=-0.10279703040334061, pairwise=0.517410071942446
- expert_id: pearson=-0.007289666779569397, spearman=0.021149204346883402, pairwise=0.517410071942446

## Conditional Diagnostics

- high_entropy pairwise=0.488862837045721, low_entropy pairwise=0.43866591294516677
- small_gap pairwise=0.4672036823935558, large_gap pairwise=0.459412780656304
- large_magnitude pairwise=0.43609022556390975, small_magnitude pairwise=0.47327044025157233

## Oracle Subset

- top32 oracle subset pairwise=0.4311717861205916, oracle_mean_delta=-0.23781204223632812, random_mean_delta=-0.014803409576416016

## Notes

- 当前 PoC 每个 group 只取消一个 expert request，不直接代表最终通信节省比例。
- 当前取消 route 代表请求暂缓后最终不释放的质量上限近似。
- routing factor diagnostics 用于控制变量看单个 routing context 因子与 realized criticality 的关系。
