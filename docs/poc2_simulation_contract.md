# POC2 Simulation Contract

## 架构

POC2.5-B 固定为两层：

- Control Plane：仅 rank0
- Data Plane：所有 GPU rank

rank0 是唯一允许执行 policy scoring、release order、release-round partition、global state update、artifact 主 summary 写入的进程。
其他 rank 只能执行收到的 `GlobalDispatchPlan`。

## Token 语义

每个 route item 至少定义：

- `route_item_index`
- `token_id`
- `token_pos`
- `route_rank`
- `origin_rank`
- `destination_rank`
- `expert_id`
- `layer_id`
- `microbatch_id`

dispatch 语义是 `origin -> destination`，return 语义是 `destination -> origin`。

## Matrix 定义

- `dispatch_bytes_matrix[source_rank][destination_rank]`
- `return_bytes_matrix[expert_rank][origin_rank]`

policy 切换后，总 dispatch/return matrix 必须不变；只允许 per-round submatrix 改变。

## GlobalDispatchPlan

每个 microbatch / policy 由 rank0 生成一个不可变 `GlobalDispatchPlan`，包含：

- `release_order`
- `release_rounds`
- `total_dispatch_matrix`
- `per_round_dispatch_matrices`
- `decision_hash`
- `scheduler_state_snapshot`

所有 rank 收到后必须重新计算 hash 并 all-gather 校验；不一致直接失败。

## State Semantics

当前采用同步模型：

- `state_semantics = historical_plus_projected`
- microbatch 完整结束后不存在真实 residual hardware queue
- completed microbatch 后：
  - `source_outbound_pending_bytes`
  - `destination_inbound_pending_bytes`
  - `flow_pending_bytes`
  - `return_flow_pending_bytes`
  必须清零
- scheduler 可见 state 由两部分构成：
  - 历史 service state：dispatch/compute/return history、hotness、latency history
  - 当前 plan projected pressure：rank0 在 build plan 时按当前 batch 未排 bucket 计算 source/destination/flow/expert pressure

## Canonical Layout

canonical segment order 固定为：

- source side：
  - `destination_rank` 升序
  - 同 destination 内按稳定 local route tuple 顺序
- destination receive side：
  - `source_rank` 升序
  - source 内按 source packing 顺序
- return side：
  - `origin_rank` 升序
  - origin 内保持 destination receive order

这是同步 release-round harness。
policy 只决定 bucket 如何组成 round，不宣称 fine-grained NIC packet scheduling，也不宣称 communication-compute overlap。

## Timing Boundary

正式 E2E 不包含：

- workload plan 构造
- payload 预生成
- metadata 预生成
- MLP weight 创建
- warmup
- artifact serialization
- Python manifest gather

正式阶段单独记录：

- payload packing
- NCCL dispatch
- destination compute
- NCCL return
- round/global sync
- `control_metadata_ms`
- `artifact_ms`
- `debug_log_ms`

主完成时间定义为：

- `global_end_to_end_completion_ms = max(local_completion_ms over ranks)`

旧 `barrier_wait_ms` 已废弃，artifact 中应读取 `barrier_wait_ms_deprecated`，不得再作为主指标。

## Correctness Mode

correctness mode 下：

- `compute_scale = 1`
- 每个 route item 对应唯一 payload row
- GPU metadata sidecar 携带：
  - `route_item_index`
  - `token_id`
  - `origin_rank`
  - `destination_rank`
  - `expert_id`
  - `route_rank`
  - `bucket_local_row_index`
- origin 必须验证 route item 恰好一次回归

任何性能 benchmark 前，必须先通过对应配置的 correctness mode。

## 当前边界

当前仍是 origin-aware、多卡 NCCL、expert-like proxy compute simulator。
它不是完整原生 MoE EP runtime，也不应直接宣称为最终端到端推理系统。
