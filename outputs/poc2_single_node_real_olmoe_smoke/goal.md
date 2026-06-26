# POC2 Single Node Scheduler Prototype

This stage upgrades proxy analysis into a runnable single-node 4-GPU scheduler prototype.

Routing mode: `real_router_logits`.
Service mode: `synthetic::synthetic-matmul`.
Current boundary: routing can be real, but service execution is still synthetic. This is not a final EP runtime and the reported latency-like metrics remain proxy metrics.
