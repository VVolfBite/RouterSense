# Router Trace Schema

Phase 0B records an observational router-logit trace, not a dispatch trace.

Each route record is a JSON object with:

- `request_id`
- `sample_id`
- `token_position`
- `layer_id`
- `expert_id`
- `expert_owner_rank` (`null` in Phase 0B)
- `expert_owner_node` (`null` in Phase 0B)
- `placement_version` (`null` in Phase 0B)
- `trace_kind` = `router_logit_observational`
- `routing_weight`
- `expert_rank_within_topk`
- `topk`
- `timestamp_ns`

The trace is observational only and reconstructs Top-K from model-returned router logits. It does not imply cross-node expert parallel semantics and does not represent a future dispatch record.
