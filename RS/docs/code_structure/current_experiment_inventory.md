# Current Experiment Inventory

Generated from current repository state.

## Experiment Scripts

| Script | Directly Runnable | Role | Status | Notes |
|---|---:|---|---|---|
| `experiments/__init__.py` | yes | internal | historical or diagnostic |  |
| `experiments/_bootstrap.py` | yes | internal | historical or diagnostic |  |
| `experiments/dev/run_gpu_validation.py` | yes | validation entry | validation |  |
| `experiments/distributed/_gpu_runner_common.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/distributed/run_gpu_a2_strategy_compare.py` | yes | internal support or legacy validation helper | historical or diagnostic | wrapped by experiments/dev/run_gpu_validation.py --suite a2 |
| `experiments/distributed/run_gpu_b2_lifecycle.py` | yes | internal support or legacy validation helper | historical or diagnostic | wrapped by experiments/dev/run_gpu_validation.py --suite b2 |
| `experiments/distributed/run_gpu_c2_async_correctness.py` | yes | internal support or legacy validation helper | historical or diagnostic | wrapped by experiments/dev/run_gpu_validation.py --suite c2 |
| `experiments/distributed/run_stage1_gloo_e2e_gate.py` | yes | validation entry | validation |  |
| `experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py` | yes | validation entry | validation |  |
| `experiments/offline/__init__.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/_timeline_prediction_diagnosis_common.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/analyze_next_layer_traffic_predictability.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/analyze_p2_consumption.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/analyze_policy_decision_timeline.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/analyze_prediction_audit.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/analyze_prediction_design_matrix.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/analyze_safe_u_decision_diff.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/build_replay_fixture_from_control_trace.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/collect_router_trace.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/estimate_planning_hiding_window.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/evaluate_fate_style_predictor.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/replay_fixture_policy_study.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/replay_online_control_trace.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/run_async_release_simulation.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_expert_to_traffic_reconstruction.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_flow_schedule_study.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_oracle_gap_replay.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_p2_bridge_async_ar0_diagnosis.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_p2_sensitivity_replay.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_prediction_oracle_baseline_closure.py` | yes | diagnostic/module wrapper | historical or diagnostic | wrapped by experiments/run_offline_replay.py |
| `experiments/offline/run_prediction_replay_suite.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_real_trace_evidence_suite.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_replay_fixture_policy_suite.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_stage1_paper_closure.py` | yes | diagnostic/module wrapper | historical or diagnostic | wrapped by experiments/run_offline_replay.py |
| `experiments/offline/run_streaming_release_simulator.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_tier1_cpu_validation.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/run_transport_stress_replay.py` | yes | diagnostic/module wrapper | historical or diagnostic |  |
| `experiments/offline/train_fate_style_predictor.py` | yes | support | historical or diagnostic |  |
| `experiments/offline/tune_existing_u_weights.py` | yes | support | historical or diagnostic |  |
| `experiments/online/__init__.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/analyze_4gpu_strategy_overhead.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/analyze_expert_to_traffic_semantics.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/analyze_full_timeline.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/audit_runtime_callgraph.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/collect_native_ep_trace.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/inventory_4gpu_artifacts.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/prepare_gpu_expert_trace_collection.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/probe_host_api.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/run_clean_megatron_forward.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/run_noop_equivalence.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/run_policy_correctness.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/run_runtime_injection_smoke.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/run_strategy_comparison.py` | yes | internal support or legacy validation helper | historical or diagnostic | wrapped by experiments/run_online_phase_sync.py and experiments/run_online_async_release.py |
| `experiments/online/support/__init__.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/support/comparison_metrics.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/support/environment_validation.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/support/phase_executor_artifacts.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/support/prepared_plan_runtime_analysis.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/support/runtime_presets.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/online/support/shadow_plan_analysis.py` | yes | internal support or legacy validation helper | historical or diagnostic |  |
| `experiments/run_offline_replay.py` | yes | formal entry | public |  |
| `experiments/run_online_async_release.py` | yes | formal entry | public |  |
| `experiments/run_online_phase_sync.py` | yes | formal entry | public |  |

## Config Files

| Config | Role | Still Used Formally |
|---|---|---|
| `configs/comparison/README.md` | legacy or specialized | no |
| `configs/comparison/default.yaml` | legacy or specialized | no |
| `configs/comparison/natural_256x128_4gpu.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_128_long.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_16x128.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_256_long.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_32x128.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_64.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_64_bucket128.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_64_bucket2048.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_64_bucket256.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_64_bucket512.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_64_long.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_8x16.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_8x32.yaml` | legacy or specialized | no |
| `configs/comparison/paper_core_4gpu_8x64.yaml` | legacy or specialized | no |
| `configs/comparison/paper_phase_local_2gpu.yaml` | legacy or specialized | no |
| `configs/comparison/paper_phase_local_2gpu_64.yaml` | legacy or specialized | no |
| `configs/comparison/paper_phase_local_2gpu_smoke.yaml` | legacy or specialized | no |
| `configs/comparison/paper_phase_local_4gpu_128_long.yaml` | legacy or specialized | no |
| `configs/comparison/paper_phase_local_4gpu_256_long.yaml` | legacy or specialized | no |
| `configs/comparison/paper_phase_local_4gpu_64_long.yaml` | legacy or specialized | no |
| `configs/comparison/tmp_comm_ramp_256x128_disabled.yaml` | legacy or specialized | no |
| `configs/comparison/tmp_comm_ramp_selected_4gpu.yaml` | legacy or specialized | no |
| `configs/comparison/tmp_comm_ramp_selected_bucket1024_4gpu.yaml` | legacy or specialized | no |
| `configs/comparison/tmp_p2_global_matrix_probe_4gpu.yaml` | legacy or specialized | no |
| `configs/comparison/tmp_prediction_probe_4gpu.yaml` | legacy or specialized | no |
| `configs/comparison/wire_slim_core_4gpu_64.yaml` | legacy or specialized | no |
| `configs/evaluation_matrix.yaml` | formal | yes |
| `configs/experiment/ablation/formal.yaml` | legacy or specialized | no |
| `configs/experiment/ablation/poc.yaml` | legacy or specialized | no |
| `configs/experiment/ablation/smoke.yaml` | legacy or specialized | no |
| `configs/experiment/offline_flow_study_4rank.yaml` | legacy or specialized | no |
| `configs/experiment/offline_policy_core_4rank.yaml` | legacy or specialized | no |
| `configs/experiment/offline_routersense_lookahead_4rank.yaml` | legacy or specialized | no |
| `configs/experiment/offline_trace_olmoe.yaml` | legacy or specialized | no |
| `configs/experiment/online_clean_megatron_local_4gpu.yaml` | legacy or specialized | no |
| `configs/experiment/online_observe_local_2gpu.yaml` | legacy or specialized | no |
| `configs/experiment/online_observe_local_4gpu_execution.yaml` | legacy or specialized | no |
| `configs/experiment/online_observe_local_4gpu_minimal.yaml` | legacy or specialized | no |
| `configs/experiment/online_policy_correctness_local_2gpu.yaml` | legacy or specialized | no |
| `configs/experiment/online_policy_debug_selected_layer.yaml` | legacy or specialized | no |
| `configs/experiment/online_policy_phase_local_2gpu.yaml` | legacy or specialized | no |
| `configs/experiment/online_policy_phase_local_4gpu.yaml` | legacy or specialized | no |
| `configs/gpu_a2_performance.yaml` | formal | yes |
| `configs/gpu_c2_correctness.yaml` | formal | yes |
| `configs/model/olmoe_1b_7b_instruct.yaml` | legacy or specialized | no |
| `configs/offline/prediction_oracle_baseline_closure.yaml` | legacy or specialized | no |
| `configs/offline/stage1_paper_closure.yaml` | legacy or specialized | no |
| `configs/offline/stage1_paper_closure_final.yaml` | legacy or specialized | no |
| `configs/offline_replay.yaml` | formal | yes |
| `configs/online_async_release.yaml` | formal | yes |
| `configs/online_phase_sync.yaml` | formal | yes |
| `configs/topology/local_2gpu.yaml` | legacy or specialized | no |
| `configs/topology/local_4gpu.yaml` | legacy or specialized | no |
| `configs/topology/two_node_2gpu.yaml` | legacy or specialized | no |
| `configs/workload/comparison_128_long_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_16x128_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_256_long_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_256x128_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_32x128_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_64_long_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_64_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_8x16_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_8x32_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_8x64_prompts.json` | legacy or specialized | no |
| `configs/workload/comparison_smoke_prompt.json` | legacy or specialized | no |
| `configs/workload/smoke_prompts.json` | legacy or specialized | no |

## Policy Names

| Canonical Name | Internal Name | Aliases | Deployable | Reference Only | Notes |
|---|---|---|---:|---:|---|
| `phase_barrier_fifo` | `phase_barrier_fifo` |  | true | false |  |
| `greedy_ready_set` | `greedy_ready_set` |  | true | false |  |
| `islip_round_robin` | `islip_round_robin` |  | true | false |  |
| `birkhoff_bucket_phase_local` | `birkhoff_phase_local` | `birkhoff_phase_local`, `B_birkhoff`, `B_birkhoff_wave` | true | false | formal deployable Birkhoff bucket baseline |
| `paired_b_barrier_criticality` | `B_barrier_criticality_matching` |  | true | false |  |
| `joint_u_barrier_criticality` | `U_barrier_criticality_global_matching` |  | true | false |  |
| `runtime_safe_u` | `RS_safe_barrier_criticality` | `safe-U` | true | false | runtime-safe-U, not posthoc best-of-U-and-B |
| `birkhoff_fluid_reference` | `birkhoff_von_neumann_fluid` | `birkhoff_von_neumann_fluid` | false | true |  |
| `exact_small_instance_oracle` | `exact_small_instance_reference` | `O_local`, `O_joint` | false | true |  |
| `posthoc_best_of_u_and_b` | `posthoc_best_of_u_and_b` | `posthoc_best_of_U_and_B` | false | true | reference only upper bound for safe selection |
