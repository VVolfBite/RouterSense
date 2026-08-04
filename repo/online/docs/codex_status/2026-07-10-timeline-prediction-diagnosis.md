# 2026-07-10 Timeline / Prediction Diagnosis

This status page summarizes the no-GPU diagnosis after the first real 4GPU collection.

Generated outputs:

- `outputs/offline/m6q_timeline_prediction_diagnosis/inventory.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/full_timeline_analysis.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/safe_u_decision_diff.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/p2_consumption_analysis.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/prediction_design_matrix.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/final_diagnosis.json`
- `outputs/offline/m6r_p2_policy_explain/expert_to_traffic_summary.json`
- `outputs/offline/m6r_p2_policy_explain/strategy_overhead_audit.json`
- `outputs/offline/m6r_p2_policy_explain/policy_decision_timeline.json`
- `outputs/offline/m6r_p2_policy_explain/barrier_criticality_p2_diagnosis.json`
- `outputs/offline/m6r_p2_policy_explain/gated_greedy_p2_diagnosis.json`
- `outputs/offline/m6r_p2_policy_explain/replay_comparison.json`
- `outputs/offline/m6r_p2_policy_explain/prediction_track_summary.json`

## Main Diagnosis

The current gap is between offline joint opportunity and the phase-sync online adapter layer.

Layered status:

- O-joint / O-local opportunity exists on small exact fixtures.
- Execution-window U replay shows about 8% gain over `B_birkhoff_wave`.
- Safe-U paired replay keeps two viable families: `RS_safe_barrier_criticality` and `RS_safe_gated_greedy`.
- Runtime-lookahead adapters are not competitive with `birkhoff_phase_local`.
- Real 4GPU phase-sync can execute `routersense_joint_priority_phase_sync`, but the current implementation is slower than Birkhoff.

## Safe-U Decision Diff

`RS_safe_barrier_criticality`:

- benefit layers: `2, 3, 6, 9, 10, 11, 12, 13, 14, 15`
- fallback layers: `4, 5, 7, 8`
- P2 changed decision count: `8`
- P2 no-effect count: `52`

`RS_safe_gated_greedy`:

- benefit layers: `2, 3, 6, 9, 10, 11, 12, 13, 14, 15`
- fallback layers: `4, 5, 7, 8, 11`
- P2 changed decision count: `7`
- P2 no-effect count: `53`

## Corrected Expert-To-Traffic Mapping

- Corrected O1 now uses:
  - aggregated `phase_context` P0 dispatch matrix
  - remote-only matrix scope
  - global expert id semantics
  - hidden-only `4096` bytes per token-assignment
- Result:
  - `o1_corrected_relative_l1 = 0.0`
  - `o1_corrected_cosine = 1.0`
  - `o1_legacy_debug_relative_l1 ≈ 0.9356`
- Conclusion:
  - expert-to-traffic deterministic mapping is semantically validated
  - the current blocker is policy P2 consumption, not trace reconstruction

## P2 Consumption

`RS_safe_barrier_criticality`:

- P2 does enter the explain trace, but aggregate influence stays small (`~0.059` on actual-trace oracle replay).
- Layer classifications are mixed:
  - `no_p2_score`
  - `p2_scale_dominated`
  - `safe_fallback_masks`
  - some `order_changed_no_bottleneck_change`
- Oracle P2 still does not improve replay.
- Most likely: P2 pressure is either too weak relative to non-P2 terms or changes order without changing the bottleneck edge.

`RS_safe_gated_greedy`:

- P2 enters with similar aggregate scale but changes order more aggressively.
- Non-oracle P2 often moves ordering in the wrong direction.
- Safe fallback masks many bad cases.
- Oracle P2 still has only a narrow positive window.

## Prediction Design

Current priority is not a more complex predictor. The next useful work is:

1. Lock expert-to-traffic semantic evaluation to real `phase_context` P0 matrices and corrected bytes.
2. Add or expose policy-level decision explanation for P2 scoring/order changes.
3. Compare traffic predictors and expert predictors only after the policy consumption path is meaningful.

## Runtime Overhead

Top online overhead targets from the real 4GPU strategy comparison:

1. `hook_before_token_dispatch_total`
2. `hook_before_token_combine_total`
3. `predict_next_dispatch`
4. `record_window_state`
5. `hook_after_token_combine_total`

Recommended order:

1. reduce/cache `predict_next_dispatch`
2. remove `prepared_phase_plan_shadow` from the hot path
3. defer/compact `record_window_state`
4. compact `store_prepared_plan`
5. re-measure actual transport makespan after control-cost reduction

Important naming correction:

- current trustworthy online measurements are inclusive hook-path durations and named control substages
- current artifacts do not support true transport makespan yet
- hook time must not be labeled `communication_makespan_us`

## Explicit Non-Claims

- `gpu_not_run=true` for this diagnosis round.
- `faithful_fate_not_validated=true`.
- `async_release_real_collectives_not_validated=true`.
- Execution-window U replay is not current online runtime evidence.
- Traffic predictor L1 improvement does not imply scheduling improvement until P2 consumption is fixed.
