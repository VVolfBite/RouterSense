# Gate 2 Closure: Cross-Layer Prediction Enables Large Pairwise Scheduling Gains

## Claim

Using cross-layer predicted next-layer traffic to drive pairwise scheduling yields a large and stable estimated communication improvement on the real OLMoE trace.

## Final evidence

Primary artifact set:

- `RS/artifacts/poc_line1/pairwise_oracle_report_batch500_v3/`

Key files:

- `pairwise_oracle_report_batch500_v3/summary.json`
- `pairwise_oracle_report_batch500_v3/results.json`
- `pairwise_oracle_report_batch500_v3/placement.json`
- `pairwise_oracle_report_batch500_v3.perf.log`

## Batch500 Gate 2 summary

From `summary.json`:

- `pair_count = 7500`
- `perfect_improvement_pct.mean = 34.2898`
- `predicted_improvement_pct.mean = 34.3392`
- `traffic_correlation.mean = 0.7280`
- `gate2_decision = PASS`

Decision rule hit:

- oracle-perfect mean improvement `>= 15%`
- oracle-predicted mean improvement `>= 10%`

Observed result:

- both are around `34%`, well above threshold

## Interpretation

This means two things at once:

1. There is substantial scheduling headroom in the pairwise `dispatch_L + combine_L + dispatch_{L+1}` window.
2. The cross-layer predictor is good enough that predicted-traffic scheduling captures essentially the same gain as perfect-traffic scheduling.

The small gap between:

- `perfect_improvement_pct.mean = 34.2898`
- `predicted_improvement_pct.mean = 34.3392`

indicates that, at this POC level, the predictor is not the limiting factor for this optimization opportunity.

## Runtime note

Final validated runtime mode:

- OR-Tools CP-SAT
- `max_time_in_seconds = 0.2`
- feasible solutions accepted
- batch500 completed in about `241.68s`

This is a POC estimation pipeline, not an exact optimizer benchmark. The point is directionality and magnitude, not proof of absolute optimality.

## Conclusion

This closes Gate 2 in the positive direction:

- cross-layer prediction can be operationalized into scheduling decisions
- the scheduling opportunity is large
- predicted traffic is sufficient to unlock that opportunity in the trace-driven pairwise analysis

## Scope note

This statement is about:

- trace-driven pairwise communication scheduling
- 4-GPU round-robin placement
- OLMoE `batch500_v2` artifacts

It is not yet a claim about:

- real distributed NCCL execution
- serving-system integration
- final production scheduler design
