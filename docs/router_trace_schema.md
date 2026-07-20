# Router Trace Schema

The current trace schema records an observational router-logit trace, not a dispatch trace.

Each route record is a JSON object with:

- `request_id`
- `sample_id`
- `token_position`
- `layer_id`
- `expert_id`
- `expert_owner_rank` (`null` for observational traces)
- `expert_owner_node` (`null` for observational traces)
- `placement_version` (`null` for observational traces)
- `trace_kind` = `router_logit_observational`
- `routing_weight`
- `expert_rank_within_topk`
- `topk`
- `timestamp_ns`

The trace is observational only and reconstructs Top-K from model-returned router logits. It does not imply cross-node expert parallel semantics and does not represent a future dispatch record.
