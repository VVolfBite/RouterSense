# Scheduling Algorithm Pairing

This document audits the historical POC1 algorithm set and rewrites it in a
pairing-first form:

- `B_*`: phase-local / independent-phase version
- `U_*`: joint / multiphase version of the same heuristic family
- `O_*`: theoretical reference only

## Table 1: B/U Paired Heuristic Families

| heuristic_family | B_phase_local_algorithm | U_joint_algorithm | B_legacy_function | U_legacy_function | B_current_status | U_current_status | paired_comparison_ready | granularity_mode | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `birkhoff_bvn` | `B_birkhoff`, `B_birkhoff_wave`, `B_barrier_aware_birkhoff`, `B_barrier_aware_birkhoff_wave` | `U_ibbr` | `fast_schedule_birkhoff`, `fast_schedule_birkhoff_wave`, `fast_schedule_barrier_aware_birkhoff`, `fast_schedule_barrier_aware_birkhoff_wave` | `fast_schedule_ibbr` | `implemented` | `recoverable` | `true` | `legacy_coarse_wave`, `legacy_fluid_reference`, `dynamic_bucket_current` | `B_birkhoff` is the strong engineering B-side baseline. `birkhoff_von_neumann_fluid` remains an oracle-like fluid sensitivity reference, but the formal paper `O_local`/`O_joint` pair now uses the same exact canonical bucket-wave runtime model and differs only by scope. `U_ibbr` is the recovered “Birkhoff seed + bottleneck GPU local swap repair” joint family candidate. |
| `gated_greedy` | `B_gated_greedy_maximal` | `U_gated_greedy_maximal`, `U_gated_greedy_maximal_atomic` | not found; derived phase-local variant | `fast_schedule_u_gated_greedy_maximal`, `fast_schedule_u_gated_greedy_maximal_atomic` | `derived_from_U` | `implemented` | `true` | `dynamic_bucket_current`, `legacy_atomic_token` | B-side is intentionally derived with the same family intent but no joint coupling, no P2, and no cross-phase release scoring. |
| `gated_maxweight_matching` | `B_gated_maxweight_matching` | `U_gated_maxweight_matching`, `U_gated_maxweight_matching_atomic` | not found; derived phase-local variant | `fast_schedule_u_gated_maxweight_matching`, `fast_schedule_u_gated_maxweight_matching_atomic` | `derived_from_U` | `implemented` | `true` | `dynamic_bucket_current`, `legacy_atomic_token` | This is the cleanest same-family B-vs-U comparison currently available in mainline. |
| `barrier_criticality_matching` | `B_barrier_criticality_matching` | `U_barrier_criticality_global_matching`, `U_barrier_criticality_global_matching_atomic` | not found; derived phase-local variant | `fast_schedule_u_barrier_criticality_global_matching`, `fast_schedule_u_barrier_criticality_global_matching_atomic` | `derived_from_U` | `implemented` | `true` | `dynamic_bucket_current`, `legacy_atomic_token` | U adds cross-phase pressure and dependency-aware urgency; B keeps only current-phase pressure scoring. |
| `barrier_price_adaptive_matching` | `B_barrier_price_adaptive_matching` | `U_barrier_price_adaptive_matching`, `U_barrier_price_adaptive_matching_atomic` | not found | `fast_schedule_u_barrier_price_adaptive_matching`, `fast_schedule_u_barrier_price_adaptive_matching_atomic` | `derived_from_U` | `recoverable` | `true` | `dynamic_bucket_current`, `legacy_atomic_token` | Historical U exists clearly. The current B-side is a formal phase-local derivation used only for fair paired comparison. |
| `lagrangian_cross_phase` | `B_lagrangian_phase_local` | `U_lagrangian` | not found | `fast_schedule_lagrangian` | `derived_from_U` | `implemented` | `true` | `legacy_atomic_token` | Historical U exists and is recovered in Tier1. The current B-side is a phase-local derivation that strips cross-phase coupling terms. |
| `cp_lpt` | `B_cp_lpt` | `U_cp_lpt` | not found | `fast_schedule_cp_lpt` | `pending` | `recoverable` | `false` | `legacy_atomic_token` | Historical U exists, but this family is not in current paired mainline until a fair B-side is defined. |
| `early_runtime_hint_adapter` | none | `routersense_p0p1p2_hint` | legacy Megatron adapter | legacy Megatron adapter | `not_found` | `implemented` | `false` | `dynamic_bucket_current` | This is an early online adapter, not the core POC1 U-family. It should not be used as the main proof of joint scheduling theory. |

## Table 2: Oracle Table

| oracle_id | implementation | scope | certified | shared model | valid objective | caveats |
| --- | --- | --- | --- | --- | --- | --- |
| `O_local_phase_oracle` | `solve_problem_exact_with_scope(..., scope="local")` | `phase_local` | `true` on supported tiny instances | `routersense_exact_bucket_wave_release_v2` | Sum of exact per-phase bucket-wave makespans | Formal paper `O_local`. Same canonical tasks/cost/replay as `O_joint`; only scope changes. Limited to at most 4 ranks and 12 canonical bucket tasks. |
| `O_joint_cp_sat_oracle` / `O_joint` | `solve_problem_exact_with_scope(..., scope="joint")` | `joint` | `true` on supported tiny instances | `routersense_exact_bucket_wave_release_v2` | Exact joint bucket-wave makespan with rank-local P0→P1→P2 release | `cp_sat` remains only a compatibility alias. Mainline no longer depends on OR-Tools or the historical atomic CP-SAT formulation. |
| `birkhoff_von_neumann_fluid` | BvN fluid decomposition | `phase_local_fluid` | deterministic reference, not the formal discrete oracle | different fluid service model | Single-phase fluid/crossbar load | Retained only as a sensitivity/reference baseline; do not place it in the same optimality-gap table as the canonical bucket-wave exact pair. |
| `legacy_atomic_cp_sat` | historical `legacy/.../scheduler/oracle.py` | `joint_atomic` | historical only | different atomic task model | Historical POC objective | Archived sensitivity result. It must not be reported as the current formal `O_joint`. |

### Oracle model contract

The formal pair shares all of the following:

- task model: `canonical_remote_edge_bucket_v1`;
- cost model: `full_duplex_matching_wave_max_rows_v1`;
- release model: `rank_local_p0_to_p1_to_p2_v1`;
- the same `TrafficInstance`, `bucket_rows`, task decomposition, wave feasibility and replay objective.

The sole experimental difference is scope:

- `O_local`: solve P0, P1 and P2 independently with phase barriers;
- `O_joint`: solve the same tasks in one dependency-released joint search space.

The historical OR-Tools CP-SAT path and the BvN fluid reference remain useful sensitivity checks, but neither is the current source of truth for the formal `O_local`/`O_joint` claim.

## Table 3: Granularity Table

| granularity_id | meaning | legacy_suffix | online_eligible | notes |
| --- | --- | --- | --- | --- |
| `legacy_coarse_wave` | legacy coarse chunk / wave decomposition | `_wave` | `false` | Historical coarse scheduling study mode. Keep only as offline ablation metadata. |
| `legacy_atomic_token` | legacy atomic/token-like or atomic-chunk scheduling | `_atomic` | `false` | Useful for offline upper-bound style studies, but not current online runtime granularity. |
| `legacy_fluid_reference` | fluid / reference-style schedule semantics | none / reference | `false` | Used for oracle-like or replay references, not direct online executor behavior. |
| `dynamic_bucket_current` | current mainline bucketized granularity | none | `true` | This is the current online-compatible granularity model. |
