# 2026-07-10 Timeline / Prediction Diagnosis

This status page summarizes the no-GPU diagnosis after the first real 4GPU collection.

Generated outputs:

- `outputs/offline/m6q_timeline_prediction_diagnosis/inventory.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/full_timeline_analysis.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/safe_u_decision_diff.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/p2_consumption_analysis.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/prediction_design_matrix.json`
- `outputs/offline/m6q_timeline_prediction_diagnosis/final_diagnosis.json`

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

## P2 Consumption

`RS_safe_barrier_criticality`:

- P2 inputs change.
- Score/order export is not yet available, so score-level diagnosis is incomplete.
- Outcome is largely insensitive to P2.
- Oracle P2 does not improve replay.
- Most likely: P2 pressure is dominated or masked before it can improve ordering.

`RS_safe_gated_greedy`:

- P2 can change outcome.
- Non-oracle P2 often moves ordering in the wrong direction.
- Oracle P2 has a narrow positive window.
- Safe fallback masks many bad cases.

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
5. re-measure communication makespan after control-cost reduction

## Explicit Non-Claims

- `gpu_not_run=true` for this diagnosis round.
- `faithful_fate_not_validated=true`.
- `async_release_real_collectives_not_validated=true`.
- Execution-window U replay is not current online runtime evidence.
- Traffic predictor L1 improvement does not imply scheduling improvement until P2 consumption is fixed.
