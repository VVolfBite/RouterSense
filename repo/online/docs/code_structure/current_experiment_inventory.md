# Current Experiment Inventory

## Public Entrypoints

- `experiments/run_offline_replay.py`: formal entrypoint
- `experiments/run_online_phase_sync.py`: formal entrypoint
- `experiments/run_online_async_release.py`: formal entrypoint

## Validation Entrypoints

- `experiments/dev/run_gpu_validation.py`: validation / gate
- `experiments/distributed/run_stage1_gloo_e2e_gate.py`: validation / gate
- `experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py`: validation / gate

## Historical / Diagnostic Entrypoints

- `experiments/__init__.py`: non-public historical or diagnostic runner
- `experiments/_bootstrap.py`: non-public historical or diagnostic runner
- `experiments/distributed/_gpu_runner_common.py`: non-public historical or diagnostic runner
- `experiments/distributed/run_gpu_a2_strategy_compare.py`: non-public historical or diagnostic runner
- `experiments/distributed/run_gpu_b2_lifecycle.py`: non-public historical or diagnostic runner
- `experiments/distributed/run_gpu_c2_async_correctness.py`: non-public historical or diagnostic runner
- `experiments/offline/__init__.py`: non-public historical or diagnostic runner
- `experiments/offline/_timeline_prediction_diagnosis_common.py`: non-public historical or diagnostic runner
- `experiments/offline/analyze_next_layer_traffic_predictability.py`: non-public historical or diagnostic runner
- `experiments/offline/analyze_p2_consumption.py`: non-public historical or diagnostic runner
- `experiments/offline/analyze_policy_decision_timeline.py`: non-public historical or diagnostic runner
- `experiments/offline/analyze_prediction_audit.py`: non-public historical or diagnostic runner
- `experiments/offline/analyze_prediction_design_matrix.py`: non-public historical or diagnostic runner
- `experiments/offline/analyze_safe_u_decision_diff.py`: non-public historical or diagnostic runner
- `experiments/offline/build_replay_fixture_from_control_trace.py`: non-public historical or diagnostic runner
- `experiments/offline/collect_router_trace.py`: non-public historical or diagnostic runner
- `experiments/offline/estimate_planning_hiding_window.py`: non-public historical or diagnostic runner
- `experiments/offline/evaluate_fate_style_predictor.py`: non-public historical or diagnostic runner
- `experiments/offline/replay_fixture_policy_study.py`: non-public historical or diagnostic runner
- `experiments/offline/replay_online_control_trace.py`: non-public historical or diagnostic runner
- `experiments/offline/run_async_release_simulation.py`: non-public historical or diagnostic runner
- `experiments/offline/run_expert_to_traffic_reconstruction.py`: non-public historical or diagnostic runner
- `experiments/offline/run_flow_schedule_study.py`: non-public historical or diagnostic runner
- `experiments/offline/run_oracle_gap_replay.py`: non-public historical or diagnostic runner
- `experiments/offline/run_p2_bridge_async_ar0_diagnosis.py`: non-public historical or diagnostic runner
- `experiments/offline/run_p2_sensitivity_replay.py`: non-public historical or diagnostic runner
- `experiments/offline/run_prediction_oracle_baseline_closure.py`: non-public historical or diagnostic runner
- `experiments/offline/run_prediction_replay_suite.py`: non-public historical or diagnostic runner
- `experiments/offline/run_real_trace_evidence_suite.py`: non-public historical or diagnostic runner
- `experiments/offline/run_replay_fixture_policy_suite.py`: non-public historical or diagnostic runner
- `experiments/offline/run_stage1_paper_closure.py`: non-public historical or diagnostic runner
- `experiments/offline/run_streaming_release_simulator.py`: non-public historical or diagnostic runner
- `experiments/offline/run_tier1_cpu_validation.py`: non-public historical or diagnostic runner
- `experiments/offline/run_transport_stress_replay.py`: non-public historical or diagnostic runner
- `experiments/offline/train_fate_style_predictor.py`: non-public historical or diagnostic runner
- `experiments/offline/tune_existing_u_weights.py`: non-public historical or diagnostic runner
- `experiments/online/__init__.py`: non-public historical or diagnostic runner
- `experiments/online/analyze_4gpu_strategy_overhead.py`: non-public historical or diagnostic runner
- `experiments/online/analyze_expert_to_traffic_semantics.py`: non-public historical or diagnostic runner
- `experiments/online/analyze_full_timeline.py`: non-public historical or diagnostic runner
- `experiments/online/audit_runtime_callgraph.py`: non-public historical or diagnostic runner
- `experiments/online/collect_native_ep_trace.py`: non-public historical or diagnostic runner
- `experiments/online/inventory_4gpu_artifacts.py`: non-public historical or diagnostic runner
- `experiments/online/prepare_gpu_expert_trace_collection.py`: non-public historical or diagnostic runner
- `experiments/online/probe_host_api.py`: non-public historical or diagnostic runner
- `experiments/online/run_clean_megatron_forward.py`: non-public historical or diagnostic runner
- `experiments/online/run_noop_equivalence.py`: non-public historical or diagnostic runner
- `experiments/online/run_policy_correctness.py`: non-public historical or diagnostic runner
- `experiments/online/run_runtime_injection_smoke.py`: non-public historical or diagnostic runner
- `experiments/online/run_strategy_comparison.py`: non-public historical or diagnostic runner
- `experiments/online/support/__init__.py`: non-public historical or diagnostic runner
- `experiments/online/support/comparison_metrics.py`: non-public historical or diagnostic runner
- `experiments/online/support/environment_validation.py`: non-public historical or diagnostic runner
- `experiments/online/support/phase_executor_artifacts.py`: non-public historical or diagnostic runner
- `experiments/online/support/prepared_plan_runtime_analysis.py`: non-public historical or diagnostic runner
- `experiments/online/support/runtime_presets.py`: non-public historical or diagnostic runner
- `experiments/online/support/shadow_plan_analysis.py`: non-public historical or diagnostic runner
