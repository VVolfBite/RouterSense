# POC1 阶段报告

## 目标

POC1 的目标是先在不改模型语义、不做真实 runtime 改造的前提下，验证 routing-derived dependency information 是否值得继续推进。

核心问题是：

- 真实 MoE routing 能不能稳定导出
- batch 级 routing 结构能不能被压缩成可分析的 bucket 特征
- dependency-aware / full 是否至少在 proxy 层面优于纯 state-only

## 已完成内容

- 真实 OLMoE trace 导出
- `batch_routing_summary`
- 多样本 `trace_batch_summary`
- `proxy_compare` 多轮迭代
- `critical_bucket_proxy` v1/v2
- prompt category grouping
- proxy 结果复核

主要入口：

- [trace_single.py](/root/autodl-tmp/RouterSense/experiment/poc1/trace_single.py)
- [trace_batch.py](/root/autodl-tmp/RouterSense/experiment/poc1/trace_batch.py)
- [proxy_compare.py](/root/autodl-tmp/RouterSense/experiment/poc1/proxy_compare.py)
- [critical_bucket_proxy.py](/root/autodl-tmp/RouterSense/experiment/poc1/critical_bucket_proxy.py)
- [verify_proxy_results.py](/root/autodl-tmp/RouterSense/experiment/poc1/verify_proxy_results.py)

## 已得到的结论

- dependency-aware 不是空想，真实 routing 中确实包含超出“单纯 bucket 大小”的结构信息
- 在 POC1 的 proxy 指标上，`dependency-only` 和 `full` 往往比 `state-only` 更接近 pseudo critical bucket
- `critical_bucket_proxy_v2` 比 v1 更稳定，也更偏向真正的 load/short-board 解释
- `layer_15` 一类中后层更值得关注，早层并不是最强信号源

## 边界

POC1 不是 runtime 证据。

它回答的是：

- routing 结构值不值得进入 runtime 假设

它没有回答：

- 多卡真实 dispatch/return 下是否真的能换来 E2E 收益

## 对后续工作的贡献

POC1 的价值是为 POC2 提供：

- 真实 routing trace
- bucket 级 dependency 特征定义
- 候选 proxy 指标
- “值得继续做执行原型”的最初证据

## 当前判断

如果只看 POC1，本来应该继续推进。
但 POC1 的结论必须被 POC2 的真实执行原型复核，不能单独当成主线定论。
