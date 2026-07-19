# Canonical Algorithm Catalog

| Canonical ID | Builder Key | Family | Deployable | Reference Only | Online Eligible | Offline Eligible | Aliases | Deprecated Aliases |
|---|---|---|---:|---:|---:|---:|---|---|
| `fifo_bucket` | `phase_barrier_fifo` | `phase_local_baseline` | `true` | `false` | `true` | `true` | `phase_barrier_fifo` | `bucketed_fifo` |
| `greedy_bucket` | `greedy_ready_set` | `phase_local_baseline` | `true` | `false` | `true` | `true` | `greedy_ready_set` | `-` |
| `islip_bucket` | `islip_round_robin` | `phase_local_baseline` | `true` | `false` | `true` | `true` | `islip_round_robin` | `-` |
| `birkhoff_bucket_phase_local` | `birkhoff_phase_local` | `phase_local_baseline` | `true` | `false` | `true` | `true` | `birkhoff_phase_local` | `-` |
| `barrier_criticality_phase_local` | `B_barrier_criticality_matching` | `phase_local_b` | `true` | `false` | `false` | `true` | `-` | `B_barrier_criticality_matching` |
| `barrier_criticality_joint` | `U_barrier_criticality_global_matching` | `joint_u` | `true` | `false` | `false` | `true` | `-` | `U_barrier_criticality_global_matching` |
| `barrier_criticality_runtime_safe` | `RS_safe_barrier_criticality` | `runtime_safe` | `true` | `false` | `false` | `true` | `runtime_safe_u` | `RS_safe_barrier_criticality, safe-U` |
| `barrier_criticality_posthoc_best` | `posthoc_best_of_u_and_b` | `reference` | `false` | `true` | `false` | `true` | `posthoc_best_of_u_and_b` | `posthoc_best_of_U_and_B` |
| `birkhoff_fluid_reference` | `birkhoff_von_neumann_fluid` | `reference` | `false` | `true` | `false` | `true` | `birkhoff_von_neumann_fluid` | `B_birkhoff, B_birkhoff_wave` |
| `oracle_local_cp_sat` | `O_local_phase_oracle` | `oracle` | `false` | `true` | `false` | `true` | `O_local_phase_oracle` | `O_local` |
| `oracle_joint_cp_sat` | `exact_small_instance_reference` | `oracle` | `false` | `true` | `false` | `true` | `O_joint_cp_sat_oracle, exact_small_instance_reference` | `O_joint, exact_small_instance_oracle` |
