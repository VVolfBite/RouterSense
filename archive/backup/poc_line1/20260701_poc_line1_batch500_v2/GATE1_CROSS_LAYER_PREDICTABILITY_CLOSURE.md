# Gate 1 Closure: Cross-Layer Prediction Is Simple and Effective on OLMoE

## Claim

On the real `OLMoE-1B-7B-0924` router trace, adjacent-layer expert prediction is strong enough to pass the Gate 1 prefetchability criterion.

## Final evidence

Primary artifact set:

- `RS/artifacts/poc_line1/full_sequence_trace_batch500_v2/`
- `RS/artifacts/poc_line1/cross_layer_report_batch500_rr_v2/`

Key files:

- `full_sequence_trace_batch500_v2/trace.jsonl`
- `full_sequence_trace_batch500_v2/hidden_states.pt`
- `full_sequence_trace_batch500_v2/gate_weights.pt`
- `cross_layer_report_batch500_rr_v2/gate1_decision.json`
- `cross_layer_report_batch500_rr_v2/layer_pair_summary.json`

## Batch500 Gate 1 decision

From `gate1_decision.json`:

- `passed = true`
- `pass_count = 13`
- `threshold.prefetch_accuracy_mean = 0.70`
- `threshold.cosine_similarity_mean = 0.70`
- `threshold.rank_correlation_mean = 0.50`

## What the layer-pair statistics show

There are 15 adjacent MoE layer pairs in the 16-layer OLMoE stack.

Observed pattern:

- early pairs `0->1`, `1->2` are weaker
- from `2->3` onward, adjacent-layer prediction is consistently strong
- many mid/late pairs reach `prefetch_accuracy_mean >= 0.80`
- cosine similarity is also high in the same region, often around `0.85-0.91`

Representative batch500 examples:

- `2->3`: prefetch `0.7157`, cosine `0.8128`
- `5->6`: prefetch `0.8049`, cosine `0.8699`
- `7->8`: prefetch `0.8736`, cosine `0.8939`
- `8->9`: prefetch `0.8837`, cosine `0.9020`
- `13->14`: prefetch `0.8730`, cosine `0.9142`

## Conclusion

This closes Gate 1 in the positive direction:

- cross-layer prediction is not fragile or prompt-specific
- it is simple to obtain from the next layer gate using current layer hidden states
- it is strong enough across most adjacent OLMoE MoE layers to support prefetch-style downstream uses

## Scope note

This statement is about:

- adjacent-layer expert prediction
- trace-driven OLMoE router behavior
- the real `batch500_v2` artifact set

It is not a claim about:

- end-to-end NCCL runtime
- scheduler optimality
- all models beyond this OLMoE configuration
