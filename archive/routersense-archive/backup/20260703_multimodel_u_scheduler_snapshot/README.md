# Archive Reason

Timestamp: 2026-07-03
Archive topic: multimodel U-scheduler snapshot

This backup is retained because it closes the current single-node validation
stage before distributed deployment:
1. it preserves the retained OLMoE execution-window comparison used for the
   current U-vs-B baseline judgment;
2. it preserves the first Qwen trace collected on the same 256-prompt corpus;
3. it preserves the first Qwen tier2 scheduler comparison and the small-sample
   fluid-oracle reference run used to assess current headroom;
4. it records the current interpretation that prediction overhead is light in
   principle, while the heavy cost today comes from the offline analysis script
   repeatedly materializing many cross-layer traffic matrices.

## Retained contents

- `artifacts/olmoe_n8_s64_summary.json`
- `artifacts/olmoe_n8_s64_table.md`
- `artifacts/olmoe_n8_s64_table_e2e.md`
- `artifacts/qwen_trace_oasst256_summary.json`
- `artifacts/qwen_tier2_s64_summary.json`
- `artifacts/qwen_tier2_s64_table.md`
- `artifacts/qwen_tier2_s64_table_e2e.md`
- `artifacts/qwen_oracle_fluid_s32_summary.json`
- `reports/20260703_multimodel_u_scheduler_snapshot.md`

## Note

Large raw trace payloads (`trace.jsonl`, `hidden_states.pt`, `gate_weights.pt`)
are intentionally not copied into this git-backed archive snapshot. The source
artifacts remain in `RS/artifacts/poc_line1/` for local reuse.
