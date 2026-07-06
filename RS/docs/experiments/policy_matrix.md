# Policy Matrix

## Formal policies

| Policy | Uses blocked P1 | Uses P2 | Offline | Online phase-local | Evaluation eligible |
|---|---:|---:|---:|---:|---:|
| `native_passthrough` | no | no | no | no | yes |
| `phase_barrier_fifo` | no | no | yes | yes | yes |
| `greedy_ready_set` | no | no | yes | yes | yes |
| `islip_round_robin` | no | no | yes | yes | yes |
| `birkhoff_phase_local` | no | no | yes | yes | yes |
| `birkhoff_von_neumann_fluid` | no | no | yes | no | yes |
| `exact_small_instance_reference` | no | no | yes | no | yes |
| `aurora_order_fixed` | no | no | yes | yes | yes |
| `fast_bvn_single_tier` | no | no | yes | yes | yes |
| `routersense_multiphase_lookahead:p0_only` | no | no | yes | no | depends on P2 source |
| `routersense_multiphase_lookahead:p0_p1` | yes | no | yes | no | depends on P2 source |
| `routersense_multiphase_lookahead:p0_p1_p2` | yes | yes | yes | no | depends on P2 source |

## Tier 1 POC-line candidates

These policies are recovered offline logical schedulers. They are not online phase-local policies and must not be resolved through `resolve_phase_policy`.

| Algorithm ID | Service model | Future information modes | Online executor compatible | Evaluation eligible |
|---|---|---|---:|---:|
| `B_birkhoff` | atomic chunk | `none`, `oracle_execution_window` | no | depends on mode/source |
| `B_birkhoff_wave` | fluid wave | `none`, `oracle_execution_window` | no | depends on mode/source |
| `U_gated_maxweight_matching` | fluid wave | `none`, `heuristic_runtime_lookahead`, `oracle_execution_window`, `oracle_predicted_runtime_lookahead` | no | depends on mode/source |
| `U_barrier_criticality_global_matching` | fluid wave | `none`, `heuristic_runtime_lookahead`, `oracle_execution_window`, `oracle_predicted_runtime_lookahead` | no | depends on mode/source |
| `U_gated_maxweight_matching_atomic` | atomic chunk | `none`, `heuristic_runtime_lookahead`, `oracle_execution_window`, `oracle_predicted_runtime_lookahead` | no | depends on mode/source |
| `U_barrier_criticality_global_matching_atomic` | atomic chunk | `none`, `heuristic_runtime_lookahead`, `oracle_execution_window`, `oracle_predicted_runtime_lookahead` | no | depends on mode/source |
| `U_lagrangian` | Lagrangian atomic chunk | `none`, `heuristic_runtime_lookahead`, `oracle_execution_window`, `oracle_predicted_runtime_lookahead` | no | depends on mode/source |

Tier 1 comparisons must be separated by service model:

- `atomic_comparison`: `B_birkhoff`, `U_gated_maxweight_matching_atomic`, `U_barrier_criticality_global_matching_atomic`.
- `fluid_comparison`: `B_birkhoff_wave`, `U_gated_maxweight_matching`, `U_barrier_criticality_global_matching`.
- `other_service_model_comparison`: `U_lagrangian` unless a later study explicitly maps it into an atomic/fluid table.

`runtime_lookahead` suppresses real P2 transport. P2 can only influence forecast pressure and diagnostics. `zero_hint` maps to `future_information_mode=none`; `copy_current_dispatch` maps to `heuristic_runtime_lookahead`; `perfect_trace` maps to `oracle_predicted_runtime_lookahead`.

`execution_window` requires actual fixture/trace P2 via `actual_trace` or compatibility name `perfect_trace`. P2 is real executable traffic in that mode, maps to `p2_role=executable_actual_traffic`, and is always `evaluation_eligible=false`.

## P2 sources

| Source | Oracle | Evaluation eligible | Notes |
|---|---:|---:|---|
| `copy_current_dispatch` | no | yes | `D_hat(l+1) = scale * D_l`, default scale `1.0`; runtime-lookahead only |
| `zero_hint` | no | yes | no future pressure |
| `shuffled_hint` | no | no | negative control only |
| `perfect_trace` | yes | no | runtime-lookahead oracle predicted pressure, or compatibility name for execution-window actual P2 |
| `actual_trace` | yes | no | execution-window actual P2 traffic only |
| `calibrated_artifact` | no for online PreparedWindowPlan hints; n/a for offline predictor artifacts | yes for online PreparedWindowPlan hints; no for unsupported offline predictor artifact | Online runtime consumes prior-layer PreparedWindowPlan hints. Offline calibrated predictor artifact remains fail-closed. |

## Online correctness suite

Selected-layer EP=2 correctness is run only for:

- `phase_barrier_fifo`
- `greedy_ready_set`
- `islip_round_robin`
- `birkhoff_phase_local`
- `aurora_order_fixed`
- `fast_bvn_single_tier`

`routersense_multiphase_lookahead:*` is excluded from online execution in this round and must fail closed.

Offline-only references are never run online:

- `birkhoff_von_neumann_fluid`: model `offline_fluid_crossbar`, not runtime-latency comparable.
- `exact_small_instance_reference`: model `discrete_bucket_phase_sync_wave`, certified only for tiny fixtures.

Online phase-local diagnostic policies `routersense_p0p1_reservation` and `routersense_p0p1p2_hint` remain distinct from formal offline RouterSense multiphase lookahead. `routersense_p0p1p2_hint` is valid for evaluating PreparedWindowPlan-derived P2 hints through the frozen phase-local executor, but it is not online multiphase joint execution.
