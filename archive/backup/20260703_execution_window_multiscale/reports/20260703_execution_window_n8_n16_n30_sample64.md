# execution_window sample64 results

## Scope

Ran only the 9 requested algorithms, with execution-window semantics (`phase2_source=actual`), `sample_limit=64`.

Algorithms:

- `B_birkhoff`
- `B_barrier_aware_birkhoff`
- `O_cp_lpt`
- `O_lagrangian`
- `O_ibbr`
- `O_gated_greedy_maximal`
- `O_gated_maxweight_matching`
- `O_barrier_criticality_global_matching`
- `O_barrier_price_adaptive_matching`

## Validation

- `pytest -q`
- result: `46 passed, 1 warning`

## Artifacts

- `RS/artifacts/poc_line1/execution_window_n8_sample64_bo/`
- `RS/artifacts/poc_line1/execution_window_n16_sample64_bo/`
- `RS/artifacts/poc_line1/execution_window_n30_sample64_bo/`

Each directory contains:

- `summary.json`
- `table.md`
- `table_e2e.md`

## Mean results

### N=8

- `B_birkhoff`: improve `18.9310%`, latency `0.5829ms`, e2e `18.9310%`
- `B_barrier_aware_birkhoff`: improve `24.2243%`, latency `6.4113ms`, e2e `24.1830%`
- `O_cp_lpt`: improve `-2.0228%`, latency `0.4797ms`, e2e `-2.0228%`
- `O_lagrangian`: improve `21.2916%`, latency `5.1717ms`, e2e `21.2305%`
- `O_ibbr`: improve `21.3108%`, latency `5.1152ms`, e2e `20.9776%`
- `O_gated_greedy_maximal`: improve `43.1025%`, latency `5.2254ms`, e2e `41.7237%`
- `O_gated_maxweight_matching`: improve `43.5457%`, latency `5.0186ms`, e2e `42.5667%`
- `O_barrier_criticality_global_matching`: improve `43.4894%`, latency `4.3585ms`, e2e `43.4894%`
- `O_barrier_price_adaptive_matching`: improve `43.4724%`, latency `7.4299ms`, e2e `42.6806%`

### N=16

- `B_birkhoff`: improve `12.4965%`, latency `1.3150ms`, e2e `12.4759%`
- `B_barrier_aware_birkhoff`: improve `17.0877%`, latency `23.1169ms`, e2e `-20.7533%`
- `O_cp_lpt`: improve `-7.8459%`, latency `1.0736ms`, e2e `-7.8459%`
- `O_lagrangian`: improve `14.2975%`, latency `6.4397ms`, e2e `13.6695%`
- `O_ibbr`: improve `13.9362%`, latency `9.3748ms`, e2e `10.0288%`
- `O_gated_greedy_maximal`: improve `40.5504%`, latency `7.8428ms`, e2e `36.4329%`
- `O_gated_maxweight_matching`: improve `40.8108%`, latency `6.4959ms`, e2e `40.7131%`
- `O_barrier_criticality_global_matching`: improve `40.7978%`, latency `8.4530ms`, e2e `35.4137%`
- `O_barrier_price_adaptive_matching`: improve `40.7911%`, latency `11.8603ms`, e2e `35.0560%`

### N=30

- `B_birkhoff`: improve `20.7120%`, latency `1.9873ms`, e2e `20.6679%`
- `B_barrier_aware_birkhoff`: improve `14.8257%`, latency `81.1473ms`, e2e `-196.1543%`
- `O_cp_lpt`: improve `0.7033%`, latency `1.5891ms`, e2e `0.6823%`
- `O_lagrangian`: improve `22.0585%`, latency `8.4950ms`, e2e `17.2503%`
- `O_ibbr`: improve `21.1194%`, latency `5.2886ms`, e2e `19.4526%`
- `O_gated_greedy_maximal`: improve `45.5846%`, latency `10.8669ms`, e2e `36.2028%`
- `O_gated_maxweight_matching`: improve `45.5118%`, latency `7.7386ms`, e2e `45.1360%`
- `O_barrier_criticality_global_matching`: improve `45.6026%`, latency `7.6585ms`, e2e `45.2929%`
- `O_barrier_price_adaptive_matching`: improve `45.5508%`, latency `14.5993ms`, e2e `32.5642%`

## Takeaways

1. `O_` global-matching methods now clearly dominate the old `B_` baselines on all three tested N values.
2. The strongest current method is `O_barrier_criticality_global_matching`.
3. `O_gated_maxweight_matching` is nearly tied, and is better than `O_barrier_criticality_global_matching` at N=16 on e2e.
4. `O_barrier_price_adaptive_matching` is competitive on raw improvement, but planning latency is still too high.
5. `O_cp_lpt` is no longer competitive in this execution-window setting.
6. `B_barrier_aware_birkhoff` becomes non-deployable once planning latency is counted.
