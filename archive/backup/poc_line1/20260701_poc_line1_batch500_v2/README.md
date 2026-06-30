# POC-line1 batch500 v2 archive

This archive marks the first 500-prompt exploration run under the corrected Gate1 semantics.

Key result:

- Gate1 `passed = true`
- `pass_count = 13`
- strong adjacent-layer prefetch accuracy persists at 500 prompts

Primary server-side paths:

- `RS/artifacts/poc_line1/full_sequence_trace_batch500_v2/`
- `RS/artifacts/poc_line1/cross_layer_report_batch500_rr_v2/`

Large raw files remain local-only and are not committed into Git history:

- `trace.jsonl`
- `hidden_states.pt`
- `gate_weights.pt`
- `prediction_rows.json`
