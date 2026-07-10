# 2026-07-10 P2 Policy Explain

This page records the no-GPU diagnosis after correcting expert-to-traffic semantics and tracing P2 through the two main safe-U families.

Primary outputs:

- `outputs/offline/m6r_p2_policy_explain/expert_to_traffic_summary.json`
- `outputs/offline/m6r_p2_policy_explain/strategy_overhead_audit.json`
- `outputs/offline/m6r_p2_policy_explain/policy_decision_timeline.json`
- `outputs/offline/m6r_p2_policy_explain/barrier_criticality_p2_diagnosis.json`
- `outputs/offline/m6r_p2_policy_explain/gated_greedy_p2_diagnosis.json`
- `outputs/offline/m6r_p2_policy_explain/replay_comparison.json`
- `outputs/offline/m6r_p2_policy_explain/prediction_track_summary.json`
- `outputs/offline/m6r_p2_policy_explain/final_diagnosis.json`

## Corrected Expert-To-Traffic Mapping

- `o1_corrected_relative_l1 = 0.0`
- `o1_corrected_cosine = 1.0`
- `o1_legacy_debug_relative_l1 ≈ 0.9356`
- actual matrix source = `phase_context_aggregated_p0_dispatch`
- bytes model = `hidden_only`
- matrix scope = `remote_only`

Conclusion:

- expert-to-traffic deterministic mapping is semantically validated
- the previous high O1 came from the wrong bytes model and the local-row debug actual matrix
- the current blocker is not trace reconstruction

## RS_safe_barrier_criticality

- mean P2 influence is small in runtime-lookahead replay
- diagnosis counts are dominated by:
  - `no_p2_score`
  - `p2_scale_dominated`
  - `safe_fallback_masks`
- oracle P2 does not improve replay

Interpretation:

- P2 often enters too weakly relative to non-P2 terms, or changes order without changing the bottleneck edge

## RS_safe_gated_greedy

- P2 is consumed more aggressively than in barrier criticality
- non-oracle P2 frequently creates `harmful_order_change`
- safe fallback masks many bad layers
- oracle P2 has only a narrow positive window

Interpretation:

- the current issue is not only predictor accuracy; it is that the policy is sensitive to wrong future pressure and lacks confidence-aware clipping

## Runtime Overhead Naming

- dispatch/combine hook timing is now reported as hook-path duration, not communication makespan
- actual transport makespan remains unavailable from current artifacts because `rank*_transport_execution.jsonl` does not expose reliable cross-rank start/end timestamps

## Current Judgment

1. Offline U opportunity still exists.
2. Corrected expert mapping is no longer the blocker.
3. Policy P2 consumption is the blocker for contribution 2.
4. Online runtime overhead is still a separate blocker for contribution 3.
5. No new candidate policy is kept as a new default in this round.

Explicit non-claims:

- `gpu_not_run=true`
- `faithful_fate_not_validated=true`
- `async_release_real_collectives_not_validated=true`
