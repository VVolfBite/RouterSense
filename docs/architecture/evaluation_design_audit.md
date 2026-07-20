# Evaluation design audit

All publishable performance rows must use
`routersense.performance_metrics.v1` and include strategy identity, baseline
identity, metric domain, time unit, trace digest and sample-set digest.

Offline logical metrics and online wall-clock metrics are distinct domains.
The formal metrics are communication makespan, P95/P99/max P1 tail latency,
first remote P1 token time, planning time, bind time, target-entry overhead,
total control time, wave count and remote P1 token count.

Strict comparisons vary one axis at a time:

- scope value: Local versus Joint with other axes fixed;
- engine value: Event versus Global with other axes fixed;
- timing value: Current versus Future with the same logical planner and input;
- prediction value: predictor changes with planner axes fixed;
- horizon value: P01/P012/P0123 with the remaining axes fixed.
