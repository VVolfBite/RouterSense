# RouteSense PoC1 Report

## Summary

- Pairwise ranking accuracy: 0.47835926449787836
- Pairwise comparisons: 3535

## Strategies

- full: mean=0.000000, median=0.000000, p95=0.000000
- random: mean=0.007788, median=0.002670, p95=0.087891
- raw_routing: mean=0.011797, median=0.000641, p95=0.085938
- oracle: mean=-0.078440, median=-0.031738, p95=-0.001129

## Notes

- 当前 PoC 每个 group 只取消一个 expert request，不直接代表最终通信节省比例。
- 当前取消 route 代表请求暂缓后最终不释放的质量上限近似。
