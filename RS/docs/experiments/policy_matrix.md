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

## P2 sources

| Source | Oracle | Evaluation eligible | Notes |
|---|---:|---:|---|
| `copy_current_dispatch` | no | yes | `D_hat(l+1) = scale * D_l`, default scale `1.0` |
| `zero_hint` | no | yes | no future pressure |
| `shuffled_hint` | no | no | negative control only |
| `perfect_trace` | yes | no | offline upper-reference only |
| `calibrated_artifact` | n/a | no | fail-closed in this round |

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

Historical diagnostic policies `routersense_p0p1_reservation` and `routersense_p0p1p2_hint` remain distinct from formal RouterSense multiphase lookahead and should not be used as production RouterSense claims.
