# POC-line1 Gate1 v2 Archive

这个归档对应 2026-07-01 的关键节点：

- Gate1 从旧的 cross-layer overlap / hit-rate 语义，修正为 Fate-style prefetch accuracy 语义
- OLMoE-1B-7B-0924-Instruct 的 router-active layer 结构被确认是 16 层
- hidden-state observational trace 已纳入主 artifact

包含内容：

- `full_sequence_trace_smoke_v2/`
- `cross_layer_report_smoke_v2/`
- `full_sequence_trace_batch32_v2/`
- `cross_layer_report_batch32_rr_v2/`
- `oracle_report_batch32_rr_v2/`

解读：

1. 旧 Gate1（overlap）失败，并不能说明跨层预测不可行。
2. 新 Gate1（prefetch accuracy）在 smoke 和 batch32 上都通过。
3. Gate2 目前仍未通过，当前真实 `oracle_vs_greedy` 改善约 `0.48%`，不能宣称存在强多层联合调度空间。

配套说明见：

- `rs_channel/reply/20260701_poc_line1_gate1_v2_status.md`
