# 2026-07-01 Dispatch-Combine Asymmetry Quick Check

## Status

Completed on the existing batch500 POC-line1 artifacts.

Output:

- `RS/artifacts/poc_line1/dc_asymmetry_report_batch500_v1/`

Archive note:

- `archive/backup/poc_line1/20260701_poc_line1_batch500_v2/DC_ASYMMETRY_NO_GO.md`

## Conclusion

This direction is currently **NO_GO**.

## Why

### Structure hypothesis

Confirmed:

- `mean asymmetry_ratio = 1.32997`
- `median = 1.3125`

So dispatch destination skew is genuinely larger than the corresponding combine-side skew.

### Scheduling hypothesis

Rejected:

- `S-greedy makespan mean = 155.28`
- `A-hotfirst makespan mean = 171.08`
- `A-hotfirst gain mean = -11.11%`

The simple asymmetric dispatch/combine heuristic is worse than the symmetric greedy baseline.

## Practical decision

Do not continue this branch for now.

The positive path remains:

1. Gate 1 cross-layer predictability
2. Gate 2 pairwise predicted scheduling

The dispatch/combine asymmetry branch does not currently add independent value.
