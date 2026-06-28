# POC2 Metric Notes

## Boundary

Current POC2 results combine:

- real routing when `routing_mode=real_router_logits`
- synthetic service execution for all current service backends

So the metrics below are still proxy metrics. They are useful for comparing scheduler behavior under the same backend, but they are not final EP runtime latency.

## Metrics

### `mean_total_batch_completion_proxy`

Mean completion time of a microbatch under the current synthetic service backend. Use it as the top-level scheduler comparison metric inside one run.

### `mean_barrier_join_proxy`

Approximate join/barrier pressure. It reflects how long the microbatch remains incomplete between earliest bucket start and final bucket finish.

### `mean_per_rank_idle_proxy_sum`

Sum of per-rank idle gaps. Lower is usually better because it suggests less starvation or imbalance across the 4 ranks.

### `mean_critical_bucket_completion_proxy`

Completion time of the bucket with the highest estimated service units. Useful for checking whether a strategy moves likely-critical work earlier.

### `mean_release_order_delta_vs_fifo`

Distance from FIFO release order. This is not a latency metric by itself; it just shows how aggressively a scheduler changes the order.
