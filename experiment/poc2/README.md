# POC2 Single-Node Scheduler Prototype

POC2 第一版已经完成统一调度器接口、4 种策略、真实 routing 接入、以及单机 4 rank 可执行原型。POC2 第二版继续推进 service execution，但边界必须说清楚。

## 当前哪些部分是真实的

- `real routing`: 在非 `--mock-routing` 模式下，runner 会真实加载本地模型并读取 `router_logits`
- `real bucket construction`: bucket、destination rank、routing-derived dependency 特征来自真实 routing
- `real scheduler decisions`: `fifo / state-only / dependency-only / full` 都在同一框架下真实给出 release/service order

## 当前哪些部分仍然是 synthetic

- service execution 还不是最终 EP runtime
- 当前 service backend 只是“更可信的 synthetic workload”，不是 expert forward path
- 所有 completion / barrier / idle / critical bucket 指标当前都还是 proxy

## 当前 service backends

- `synthetic-matmul`: GPU matmul/relu workload，成本显式依赖 `token_count` 和 `estimated_service_units`
- `synthetic-token-linear`: GPU linear stack workload，更像 token-batch 线性层堆叠
- `mock-sleep`: CPU sleep backend，适合无卡烟测

## 四种策略

- `fifo`: 按到达顺序
- `state-only`: 只看 runtime-visible state
- `dependency-only`: 只看 routing-derived structure
- `full`: 组合 state 和 dependency

## 常用入口

无卡烟测：

```bash
python experiment/poc2/single_node_runner.py \
  --model olmoe \
  --mock-routing \
  --service-backend mock-sleep \
  --microbatch-size 2 \
  --microbatch-count 2 \
  --output-dir outputs/poc2_single_node_eval_smoke
```

真实 routing + synthetic GPU service：

```bash
python experiment/poc2/single_node_runner.py \
  --model olmoe \
  --service-backend synthetic-matmul \
  --output-dir outputs/poc2_single_node_real_olmoe
```

## 当前结果怎么看

先看：

- `scheduler_decision.release_order`
- `scheduler_decision.rank_service_order`
- `mean_total_batch_completion_proxy`
- `mean_barrier_join_proxy`
- `mean_per_rank_idle_proxy_sum`
- `mean_critical_bucket_completion_proxy`
- `mean_release_order_delta_vs_fifo`

## 下一步目标

下一步不是再刷 proxy，而是继续向“更接近真实 service path”的单机 runtime 推进；但本阶段仍不假装已经实现完整 EP runtime。
