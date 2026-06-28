# POC2 Single-Node Scheduler Prototype

POC2 第一版已经完成统一调度器接口、4 种策略、真实 routing 接入、以及单机 4 rank 可执行原型。POC2 第二版继续推进 service execution，但边界必须说清楚。
POC2.5 的重点不是性能，而是修复并验证真正的 `origin -> destination -> origin` NCCL dispatch/return 语义。

## 当前哪些部分是真实的

- `real routing`: 在非 `--mock-routing` 模式下，runner 会真实加载本地模型并读取 `router_logits`
- `real bucket construction`: bucket、destination rank、routing-derived dependency 特征来自真实 routing
- `origin-aware dispatch semantics`: POC2.5 之后，WorkloadPlan 显式保留 `origin_rank / destination_rank / token_id / route_id`
- `real NCCL dispatch/return`: 单机多卡路径使用真实 GPU↔GPU `torch.distributed.all_to_all_single`
- `real scheduler decisions`: `fifo / state-only / dependency-only / full` 都在同一框架下真实给出 release/service order

## 当前哪些部分仍然是 synthetic

- local expert compute 还不是最终模型 expert forward
- expert placement 仍是受控 mapping，不等于生产部署
- 因此当前 harness 仍不是完整端到端 MoE inference runtime

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

4 卡固定远端流 correctness integration：

```bash
torchrun --standalone --nproc_per_node=4 \
  experiment/poc2/verify_remote_dispatch_4gpu.py
```

真实 router trace semantic smoke：

```bash
torchrun --standalone --nproc_per_node=4 \
  experiment/poc2/verify_real_trace_semantics.py \
  --model olmoe \
  --placement-policy balanced \
  --origin-sharding round-robin
```

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

下一步必须先以 POC2.5 的 artifact 复核通信语义，再决定是否恢复 scheduler benchmark。未复核前，不应再引用旧 POC2 NCCL 结果做性能结论。
