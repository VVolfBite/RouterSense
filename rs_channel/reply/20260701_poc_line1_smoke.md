# POC-line1 Smoke Report

## Status

- commit: `dcf85d8e0fc015e9f63b4de9bf029e0b87322785`
- scope: single-GPU real OLMoE full-sequence trace + offline Gate 1 / Gate 2 smoke
- model path: `/root/autodl-tmp/models/OLMoE-1B-7B-0924`

## Key Findings

1. Real full-sequence trace collection succeeded.
2. Real checkpoint returned `16` router layers, not `4`.
3. Gate 1 failed on this smoke run: cross-layer top-k overlap is too low.
4. Gate 2 passed in the current offline oracle-vs-greedy simulator.
5. Therefore, current evidence says:
   - multi-layer scheduling room exists in the offline model
   - but the predictive premise is currently weak on real trace

## Real Trace Smoke

Artifact path:
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/full_sequence_trace_smoke/`
- backup copy: `/root/autodl-tmp/RouterSense/archive/backup/poc_line1/full_sequence_trace_smoke/`

Summary:
- `moe_layer_count = 16`
- `moe_layer_ids = [0..15]`
- `topk = 8`
- `token_count = 19`
- `record_count = 2432`

## Gate 1

Artifact path:
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/cross_layer_report_smoke/`
- backup copy: `/root/autodl-tmp/RouterSense/archive/backup/poc_line1/cross_layer_report_smoke/`

Decision:
- `passed = false`
- `pass_count = 0`
- `rank_pass = true`

Representative pair:
- `0->1 mean hit_rate = 0.1118`
- `0->1 mean weighted_hit_rate = 0.0361`

Interpretation:
- adjacent-layer expert overlap is far below the current Gate 1 threshold
- this weakens the current cross-layer prediction thesis

## Gate 2

Artifact path:
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/oracle_report_smoke/`
- backup copy: `/root/autodl-tmp/RouterSense/archive/backup/poc_line1/oracle_report_smoke/`

Decision:
- `mean_improvement_pct = 35.0`
- `passed = true`

Representative numbers:
- `greedy_makespan = 2671.0`
- `oracle_makespan = 1736.15`

Interpretation:
- if future traffic were known, the current offline model still has meaningful scheduling space
- but Gate 1 shows the predictor itself is not yet validated

## Current Recommendation

Next useful step is not more scheduler code.
Next useful step is a larger real-trace batch:
- multiple prompts
- same analysis pipeline
- compare whether Gate 1 failure is stable or prompt-specific

