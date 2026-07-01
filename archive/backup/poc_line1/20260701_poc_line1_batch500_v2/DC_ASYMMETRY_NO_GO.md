# Dispatch-Combine Asymmetry Quick Check: NO_GO

## Question

We tested whether dispatch/combine structural asymmetry can support an additional scheduling gain beyond the current pairwise scheduling line.

Two hypotheses were checked:

1. Structural hypothesis:
   dispatch destination skew is materially larger than the corresponding combine-side source skew.
2. Scheduling hypothesis:
   using a differentiated dispatch/combine strategy improves makespan beyond a symmetric greedy baseline.

## Artifact

- `RS/artifacts/poc_line1/dc_asymmetry_report_batch500_v1/`

Key files:

- `asymmetry_summary.json`
- `layer_asymmetry_summary.json`
- `dispatch_combine_scatter.json`
- `schedule_summary.json`
- `go_no_go.json`

## Result

### Part A: Structural asymmetry

Confirmed.

From `asymmetry_summary.json`:

- `mean asymmetry_ratio = 1.32997`
- `median = 1.3125`
- `p25 = 1.2083`
- `p75 = 1.4167`
- `max = 1.875`

So dispatch/combine is not symmetric in structure. The skew difference is real but moderate, not extreme.

### Part B: Fast differential scheduling

Rejected.

From `schedule_summary.json`:

- `S-greedy makespan mean = 155.28`
- `A-hotfirst makespan mean = 171.08`
- `A-hotfirst gain mean = -11.11%`

The dispatch-hotfirst / combine-balanced heuristic is worse than the symmetric greedy baseline on this dataset.

## Interpretation

This gives a clean negative result:

- the structural asymmetry exists
- but this simple differentiated scheduling rule does not convert that asymmetry into extra benefit

So the current direction is **NO_GO** at this stage.

## Decision

From `go_no_go.json`:

- `asymmetry_significant = true`
- `dispatch_combine_independent_value = false`
- `decision = NO_GO`

## Practical conclusion

Do not prioritize a separate dispatch/combine asymmetry scheduling line right now.

The pairwise cross-layer scheduling result already shows large gains, and this extra asymmetry-specific heuristic does not add value in the quick validation.
