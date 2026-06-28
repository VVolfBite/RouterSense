# POC2 Correctness Audit

本轮目标不是性能结论，而是修复并审计 POC2 从 router trace 到 NCCL dispatch/return 的通信语义。

## 核心发现

1. 旧实现是 `destination-centric`。
当前 `WorkloadPlan` 只按 `destination_rank` 聚合 bucket，没有 `origin_rank`、`token_id`、`route_id`。

2. 旧 dispatch 语义错误。
运行时每个 rank 过滤 `bucket.destination_rank == current_rank` 后，本地直接生成 payload，再通过 NCCL 发送。这意味着目标 rank 在“发送给自己应处理的 bucket”，而不是 origin rank 把自己持有的 token 发往目标 rank。

3. 旧 return 语义也随之错误。
metadata 中的 `origin_rank` 来自当前 rank，而不是 token 原始 owner，因此 return 并不保证回到真实 origin。

4. 这会误导所有通信结论。
即使看到了 all-to-all 调用，也可能主要是 self-send / 对角流，不能作为真实 MoE source→destination dispatch 的证据。

## 修复后的语义

1. `runner.py` 现在保留 route-item 级别的 `route_id / token_id / token_pos / expert_id / microbatch_id`。
2. `distributed_runtime.py` 的 `WorkloadBucket` 现在包含：
   `origin_rank / destination_rank / expert_id / token_ids / route_id / payload_bytes`。
3. `WorkloadPlan` 按 `(origin_rank, destination_rank, expert_id)` 聚合。
4. 每个 rank 只 materialize 自己 `origin_rank == current_rank` 的 payload。
5. dispatch matrix 定义为 `dispatch_bytes[source_rank][destination_rank]`。
6. return matrix 定义为 `return_bytes[expert_rank][origin_rank]`。
7. destination-side compute 只对“收到的 payload”执行。
8. return 按收到 metadata 中的真实 `origin_rank` 返回。

## 仍然不是最终 runtime 的边界

1. router trace 是真实的。
2. expert placement 仍是受控 placement，不是原模型生产部署。
3. local compute 仍是 expert-like proxy MLP，不是原模型 expert 权重。
4. 本轮不输出任何 scheduler 性能结论，只验证语义正确性。
