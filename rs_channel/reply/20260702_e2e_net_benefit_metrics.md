# End-to-End Net-Benefit Metrics

Date: 2026-07-02

Artifacts:
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/e2e_net_benefit_n8_v2/summary.json`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/e2e_net_benefit_n8_v2/table.md`
- `/root/autodl-tmp/RouterSense/RS/artifacts/poc_line1/e2e_net_benefit_n8_v2/table_e2e.md`

Validation:
- `pytest -q tests/test_poc_line1.py` → `19 passed`

Parameters:
- `hidden_window_ms = 10.0`
- `token_to_ms_factor = 0.5`

Added metrics:
- `comm_savings_ms`
- `exposed_latency_ms`
- `net_benefit_ms`
- `effective_e2e_time_ms`
- `e2e_speedup_pct`
- `effective_latency_ms`

Core N=8 end-to-end table:

| algorithm | greedy_mksp | algo_mksp | comm_savings_ms | sched_latency_ms | exposed_ms | net_benefit_ms | e2e_time_ms | e2e_speedup% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy (no sched) | 65.9167 | 65.9167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 65.9167 | 0.0000 |
| birkhoff | 65.9167 | 49.8208 | 16.0958 | 2.5277 | 0.0000 | 16.0958 | 49.8208 | 22.9499 |
| ibbr | 65.9167 | 49.5750 | 16.3417 | 6.3628 | 0.1054 | 16.2362 | 49.6804 | 23.1441 |
| ejection_chain_tabu | 65.9167 | 47.2667 | 18.6500 | 29.4228 | 19.4516 | -0.8016 | 66.7182 | -7.8840 |
| lns_cp_repair | 65.9167 | 47.2667 | 18.6500 | 19.3480 | 9.3787 | 9.2713 | 56.6453 | 10.5081 |
| cp_lpt | 65.9167 | 61.4042 | 4.5125 | 1.2317 | 0.0000 | 4.5125 | 61.4042 | 5.4120 |
| lagrangian | 65.9167 | 48.8417 | 17.0750 | 5.7620 | 0.0000 | 17.0750 | 48.8417 | 24.5480 |
| oracle_perfect | 65.9167 | 34.5708 | 31.3458 | 111.8335 | 104.3227 | -72.9769 | 138.8935 | -129.0243 |

Sanity checks:
- `greedy` net benefit is exactly zero.
- `birkhoff` has positive net benefit.
- `oracle_perfect` has strongly negative net benefit under a 10ms hidden window because solve time dominates communication savings.

Interpretation:
- `lagrangian` is currently the strongest deployable candidate in this setting: high savings with fully hidden planning cost.
- `birkhoff` remains a very strong low-overhead baseline.
- `ejection_chain_tabu` is a good example of why isolated makespan improvement is insufficient: its raw schedule is better than `birkhoff`, but exposed planner cost wipes out the gain.
- `oracle_perfect` is valuable as an upper-bound oracle only, not as a practical online scheduler.
