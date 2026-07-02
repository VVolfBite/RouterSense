# Archive Reason

Timestamp: 2026-07-03
Archive topic: oracle / fast / greedy optimization-gap study

This backup is retained because it establishes the main theoretical motivation:
1. oracle remains materially better than greedy / strong fast baselines;
2. the optimization headroom is on the order of 20%-40% depending on setup;
3. CP-SAT performance logs are kept with the report for later audit.

## Retained contents
- artifacts/pairwise_fast_report_batch500_rr_v3
- artifacts/pairwise_oracle_report_batch500_v3
- artifacts/pairwise_oracle_report_batch500_v3.perf.log
- reports/20260702_fast_scheduler_batch500_result.md
- docs/poc_line1_status.md
- docs/multiphase_global_matching_study.md

## Note

The very large raw `results.json` payloads are intentionally excluded from git.
This archive keeps the summary snapshots, strategy summary, placement metadata,
perf log, and the report needed for controlled review.
