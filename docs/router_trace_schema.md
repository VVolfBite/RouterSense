# Router Trace Schema

Each route record is a JSON object with:

- `request_id`
- `sample_id`
- `token_position`
- `layer_id`
- `expert_id`
- `expert_rank`
- `routing_weight`
- `expert_rank_within_topk`
- `topk`
- `timestamp_ns`

The trace is observational only and does not imply cross-node expert parallel semantics.

