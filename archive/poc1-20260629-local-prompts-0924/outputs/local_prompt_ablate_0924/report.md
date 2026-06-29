# RouteSense PoC1 Report

## Summary

- Pairwise ranking accuracy: 0.49390243902439024
- Pairwise comparisons: 656

## Strategies

- full: mean=0.000000, median=0.000000, p95=0.000000
- random: mean=0.008811, median=0.006073, p95=0.103027
- raw_routing: mean=-0.008931, median=0.000000, p95=0.031250
- oracle: mean=-0.129664, median=-0.062866, p95=-0.007812

## Notes

- 当前 PoC 每个 group 只取消一个 expert request，不直接代表最终通信节省比例。
- 当前取消 route 代表请求暂缓后最终不释放的质量上限近似。
