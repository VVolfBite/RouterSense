# Algorithm Name Migration

| 旧名称 | 正式名称 | 当前语义 | 是否仍在使用 | 处理方式 |
|---|---|---|---|---|
| `bucketed_fifo` | `fifo_bucket` | `deployable_bucket` | 是 | bridge |
| `runtime_safe_u` | `barrier_criticality_runtime_safe` | `deployable_bucket` | 是 | bridge |
| `safe-U` | `barrier_criticality_runtime_safe` | `deployable_bucket` | 是 | bridge |
| `posthoc_best_of_U_and_B` | `barrier_criticality_posthoc_best` | `posthoc_reference` | 是 | alias/reference |
| `B_birkhoff` | `birkhoff_fluid_reference` | `fluid_reference` | 是 | alias/reference |
| `B_birkhoff_wave` | `birkhoff_fluid_reference` | `fluid_reference` | 是 | alias/reference |
| `O_local` | `oracle_local_cp_sat` | `exact_reference` | 是 | alias/reference |
| `O_joint_cp_sat_oracle` | `oracle_joint_cp_sat` | `exact_reference` | 是 | alias/reference |
| `O_joint` | `oracle_joint_cp_sat` | `exact_reference` | 是 | alias/reference |
| `exact_small_instance_oracle` | `oracle_joint_cp_sat` | `exact_reference` | 是 | alias/reference |
