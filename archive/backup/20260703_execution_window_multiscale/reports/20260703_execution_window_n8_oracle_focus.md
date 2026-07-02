# execution_window N=8 oracle focus

## Scope

- mode: `execution_window`
- `num_gpus=8`
- `sample_limit=16`
- `phase2_source=actual`
- compared only:
  - `B_birkhoff`
  - `O_lagrangian`
  - `O_gated_maxweight_matching`
  - `O_barrier_criticality_global_matching`
  - `oracle_perfect`

Artifacts:

- `RS/artifacts/poc_line1/execution_window_n8_sample16_oracle_focus/summary.json`
- `RS/artifacts/poc_line1/execution_window_n8_sample16_oracle_focus/table.md`
- `RS/artifacts/poc_line1/execution_window_n8_sample16_oracle_focus/table_e2e.md`

## Mean results

- `B_birkhoff`: improve `20.7261%`, latency `2.4652ms`, e2e `20.7261%`
- `O_lagrangian`: improve `21.2905%`, latency `6.6572ms`, e2e `21.1412%`
- `O_gated_maxweight_matching`: improve `44.8612%`, latency `18.8895ms`, e2e `29.5383%`
- `O_barrier_criticality_global_matching`: improve `44.8300%`, latency `15.1029ms`, e2e `35.7070%`
- `oracle_perfect`: improve `45.3514%`, latency `120.2307ms`, e2e `-169.0736%`

## Oracle status note

The current oracle still uses the 200ms time cap, so its per-pair solver statuses are mixed:

- many pairs: `OPTIMAL`
- many harder pairs: `FEASIBLE`

So in this run oracle should be treated as a strong upper bound / reference, not as a guaranteed exact optimum for every pair.

## Main takeaway

On this N=8 slice, `O_barrier_criticality_global_matching` is nearly matching oracle-level scheduling gain while remaining far cheaper than oracle in planning cost.
