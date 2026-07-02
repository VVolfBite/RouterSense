## 2026-07-02 Fast Scheduler Batch500 Result

- code head: `0c642b1`
- result path: `RS/artifacts/poc_line1/pairwise_fast_report_batch500_rr_v3/`
- summary file:
  - `RS/artifacts/poc_line1/pairwise_fast_report_batch500_rr_v3/summary.json`

### Batch500 summary

- pair_count: `7500`
- gate2_decision: `PASS`
- oracle_perfect mean improvement: `34.2888%`
- oracle_predicted mean improvement: `34.3321%`
- traffic_correlation mean: `0.7280`

### Fast candidate summary

- fast_schedule_pairwise:
  - mean improvement: `9.2899%`
  - mean solve time: `1.1262 ms`
- lookahead_lpt:
  - mean improvement: `-4.4896%`
  - mean solve time: `0.0987 ms`
- cp_lpt:
  - mean improvement: `-1.5949%`
  - mean solve time: `0.1117 ms`
- birkhoff:
  - mean improvement: `1.4489%`
  - mean solve time: `0.2510 ms`
- iterated_greedy:
  - mean improvement: `29.0594%`
  - mean solve time: `14.0818 ms`
- cp_local_swap:
  - mean improvement: `6.8485%`
  - mean solve time: `1.7679 ms`

### Interpretation

- current best-quality fast candidate is still `iterated_greedy`, but it is much slower than the sub-5ms target
- current best low-latency aggregate selector is `fast_schedule_pairwise`, with about `9.29%` mean improvement at about `1.13 ms`
- oracle gap remains large: there is still significant scheduling headroom beyond current fast heuristics
