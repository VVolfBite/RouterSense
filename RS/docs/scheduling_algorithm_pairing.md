# Scheduling Algorithm Pairing

This document audits the historical POC1 algorithm set and rewrites it in a
pairing-first form:

- `B_*`: phase-local / independent-phase version
- `U_*`: joint / multiphase version of the same heuristic family
- `O_*`: theoretical reference only

## Table 1: B/U Paired Heuristic Families

| heuristic_family | B_phase_local_algorithm | U_joint_algorithm | B_legacy_function | U_legacy_function | B_current_status | U_current_status | paired_comparison_ready | granularity_mode | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `birkhoff_bvn` | `B_birkhoff`, `B_birkhoff_wave`, `B_barrier_aware_birkhoff`, `B_barrier_aware_birkhoff_wave` | `U_ibbr` | `fast_schedule_birkhoff`, `fast_schedule_birkhoff_wave`, `fast_schedule_barrier_aware_birkhoff`, `fast_schedule_barrier_aware_birkhoff_wave` | `fast_schedule_ibbr` | `implemented` | `recoverable` | `true` | `legacy_coarse_wave`, `legacy_fluid_reference`, `dynamic_bucket_current` | `B_birkhoff` is also the local oracle-like reference. `U_ibbr` is now recovered as the historical “Birkhoff seed + bottleneck GPU local swap repair” joint family candidate, so `birkhoff_bvn` is back in the paired mainline. |
| `gated_greedy` | `B_gated_greedy_maximal` | `U_gated_greedy_maximal`, `U_gated_greedy_maximal_atomic` | not found; derived phase-local variant | `fast_schedule_u_gated_greedy_maximal`, `fast_schedule_u_gated_greedy_maximal_atomic` | `derived_from_U` | `implemented` | `true` | `dynamic_bucket_current`, `legacy_atomic_token` | B-side is intentionally derived with the same family intent but no joint coupling, no P2, and no cross-phase release scoring. |
| `gated_maxweight_matching` | `B_gated_maxweight_matching` | `U_gated_maxweight_matching`, `U_gated_maxweight_matching_atomic` | not found; derived phase-local variant | `fast_schedule_u_gated_maxweight_matching`, `fast_schedule_u_gated_maxweight_matching_atomic` | `derived_from_U` | `implemented` | `true` | `dynamic_bucket_current`, `legacy_atomic_token` | This is the cleanest same-family B-vs-U comparison currently available in mainline. |
| `barrier_criticality_matching` | `B_barrier_criticality_matching` | `U_barrier_criticality_global_matching`, `U_barrier_criticality_global_matching_atomic` | not found; derived phase-local variant | `fast_schedule_u_barrier_criticality_global_matching`, `fast_schedule_u_barrier_criticality_global_matching_atomic` | `derived_from_U` | `implemented` | `true` | `dynamic_bucket_current`, `legacy_atomic_token` | U adds cross-phase pressure and dependency-aware urgency; B keeps only current-phase pressure scoring. |
| `barrier_price_adaptive_matching` | `B_barrier_price_adaptive_matching` | `U_barrier_price_adaptive_matching`, `U_barrier_price_adaptive_matching_atomic` | not found | `fast_schedule_u_barrier_price_adaptive_matching`, `fast_schedule_u_barrier_price_adaptive_matching_atomic` | `derived_from_U` | `recoverable` | `true` | `dynamic_bucket_current`, `legacy_atomic_token` | Historical U exists clearly. The current B-side is a formal phase-local derivation used only for fair paired comparison. |
| `lagrangian_cross_phase` | `B_lagrangian_phase_local` | `U_lagrangian` | not found | `fast_schedule_lagrangian` | `derived_from_U` | `implemented` | `true` | `legacy_atomic_token` | Historical U exists and is recovered in Tier1. The current B-side is a phase-local derivation that strips cross-phase coupling terms. |
| `cp_lpt` | `B_cp_lpt` | `U_cp_lpt` | not found | `fast_schedule_cp_lpt` | `pending` | `recoverable` | `false` | `legacy_atomic_token` | Historical U exists, but this family is not in current paired mainline until a fair B-side is defined. |
| `early_runtime_hint_adapter` | none | `routersense_p0p1p2_hint` | legacy Megatron adapter | legacy Megatron adapter | `not_found` | `implemented` | `false` | `dynamic_bucket_current` | This is an early online adapter, not the core POC1 U-family. It should not be used as the main proof of joint scheduling theory. |

## Table 2: Oracle Table

| oracle_id | implementation | scope | deterministic_solver | heavy_solver | valid_objective | caveats |
| --- | --- | --- | --- | --- | --- | --- |
| `O_local_phase_oracle` | `B_birkhoff` / legacy `fast_schedule_birkhoff` | `phase_local` | `true` | `false` | Single-phase fluid / crossbar makespan reference | Oracle-like only for local fluid makespan semantics. Not guaranteed optimal for wave count or setup-heavy metrics. |
| `O_joint_cp_sat_oracle` | legacy `pairwise_oracle` in `legacy/historical_poc/src_rs_legacy/scheduler/oracle.py` | `joint` | `true` when CP-SAT solves exactly within budget; otherwise bounded/feasible` | `true` | Joint P0/P1/P2 coupled objective under solver formulation | Historical code exposes `objective`, `best_bound`, `optimality_gap`, and time limit, but also has a greedy fallback path when OR-Tools is unavailable. |
| `exact_small_instance_reference` | `exact_small_instance_reference` | `joint_small_instance` | `true` | `true` | Very small exact formal instances | Not a scalable main oracle. Used as a small-instance correctness reference only. |

### CT oracle audit note

During this audit, no separate legacy symbol literally named `CT oracle` was found.
The closest concrete joint-oracle implementation in historical POC1 is:

- `legacy/historical_poc/src_rs_legacy/scheduler/oracle.py::pairwise_oracle`

That implementation is CP-SAT based when OR-Tools is available, carries
`objective`, `best_bound`, and `optimality_gap`, and falls back to a greedy upper
bound path only when the solver backend is unavailable. For current mainline
documentation, this is the safest concrete representation of the remembered joint
oracle concept.

## Table 3: Granularity Table

| granularity_id | meaning | legacy_suffix | online_eligible | notes |
| --- | --- | --- | --- | --- |
| `legacy_coarse_wave` | legacy coarse chunk / wave decomposition | `_wave` | `false` | Historical coarse scheduling study mode. Keep only as offline ablation metadata. |
| `legacy_atomic_token` | legacy atomic/token-like or atomic-chunk scheduling | `_atomic` | `false` | Useful for offline upper-bound style studies, but not current online runtime granularity. |
| `legacy_fluid_reference` | fluid / reference-style schedule semantics | none / reference | `false` | Used for oracle-like or replay references, not direct online executor behavior. |
| `dynamic_bucket_current` | current mainline bucketized granularity | none | `true` | This is the current online-compatible granularity model. |
