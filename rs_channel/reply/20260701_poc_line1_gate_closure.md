# 2026-07-01 POC-line1 Gate Closure

## Final status

POC-line1 now has two closed experimental claims:

1. Cross-layer prediction on OLMoE is simple and effective.
2. That prediction signal can drive pairwise scheduling with large estimated communication improvement.

## Preserved artifacts

Gate 1:

- `RS/artifacts/poc_line1/full_sequence_trace_batch500_v2/`
- `RS/artifacts/poc_line1/cross_layer_report_batch500_rr_v2/`

Gate 2:

- `RS/artifacts/poc_line1/pairwise_oracle_report_batch500_v3/`

## Archived closure docs

- `archive/backup/poc_line1/20260701_poc_line1_batch500_v2/GATE1_CROSS_LAYER_PREDICTABILITY_CLOSURE.md`
- `archive/backup/poc_line1/20260701_poc_line1_batch500_v2/GATE2_PAIRWISE_SCHEDULING_CLOSURE.md`

## Final numbers

Gate 1:

- `passed = true`
- `pass_count = 13 / 15 adjacent layer pairs`

Gate 2:

- `pair_count = 7500`
- `perfect_improvement_pct.mean = 34.2898`
- `predicted_improvement_pct.mean = 34.3392`
- `gate2_decision = PASS`

## Repo cleanup

Removed the obsolete pre-pairwise oracle path:

- deleted `RS/experiments/poc_line1/oracle_vs_greedy.py`
- removed old multi-layer oracle helper exports and tests

The active POC-line1 scheduling path is now the validated cross-layer + pairwise path only.
