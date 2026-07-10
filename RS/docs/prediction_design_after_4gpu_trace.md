# Prediction Design After 4GPU Trace

This document separates three concerns that were previously easy to conflate.

## 1. Expert Prediction

Definition:

- Input: current-layer `source_rank x expert_id` counts or router/gate data.
- Output: next-layer `source_rank x expert_id` counts.
- Mapping: use `expert_to_rank_map` to reconstruct rank-to-rank traffic.

Current status:

- Real 4GPU expert trace collection succeeded.
- `source_expert_counts` are non-empty for 4 ranks across 16 layers.
- World merge succeeds with no missing or conflicting source ranks.
- The original O1 reconstruction report showed high L1, but the semantic audit found that the old target/bytes path was misleading.
- Corrected O1 now uses:
  - actual matrix source = aggregated `phase_context` P0 dispatch matrix
  - matrix scope = remote-only
  - expert id semantic = global expert id
  - bytes model = hidden-only `4096` bytes per token-assignment
- Under that corrected path, `o1_corrected_relative_l1 = 0.0`.
- The legacy local-row audit path remains useful only as a debug artifact; its mean relative L1 stays around `0.9356` and must not be used for paper claims.

Candidate expert predictors:

- `source_rank_expert_copy`
- `source_rank_expert_transition`
- `layer_pair_expert_transition`
- source-rank-conditioned transition
- faithful gate replay

Do not claim faithful FATE until router/gate replay is implemented and validated.

## 2. Traffic Prediction

Definition:

- Input: current or historical rank-to-rank traffic matrix.
- Output: next-layer rank-to-rank traffic matrix.

Current evidence:

- `copy_current_dispatch` is a useful traffic baseline.
- `fate_style_history` is currently the best non-oracle traffic baseline in the replay summary:
  - relative L1 around `0.2236`
  - cosine around `0.8974`
- `fate_style_linear` is not consistently better.

These are traffic-matrix baselines, not faithful FATE expert predictors.

## 3. Policy Consumption

Even a good predictor does not help if the scheduler does not consume P2 correctly.

Current observations:

- `RS_safe_barrier_criticality` is largely insensitive to P2.
- `RS_safe_gated_greedy` reacts to P2, but non-oracle P2 often makes replay worse.
- Oracle P2 has only a narrow positive window in the current safe-U policies.

Therefore, the next optimization should focus on policy P2 consumption before adding predictor complexity.

Current P2-consumption diagnosis:

- `RS_safe_barrier_criticality`
  - P2 enters the scheduler, but its aggregate influence is small (`~0.059` in the current explain trace aggregate).
  - Most layers fall into `no_p2_score`, `p2_scale_dominated`, or `safe_fallback_masks`.
  - Oracle P2 still does not improve replay.
- `RS_safe_gated_greedy`
  - P2 also enters with similar aggregate scale (`~0.057`), but the policy is more order-sensitive.
  - Non-oracle P2 frequently produces `harmful_order_change`.
  - Safe fallback saves many bad layers, but oracle P2 still shows only a narrow positive window.

## Recommended Sequence

1. Keep the corrected expert-to-traffic semantic path fixed.
2. Use debug-only policy decision explanations to inspect where P2 changes score/order and where it gets masked.
3. Repair P2 consumption in `RS_safe_barrier_criticality` and `RS_safe_gated_greedy`.
4. Compare traffic predictors against expert predictors only after policy consumption is meaningful.
5. Only then implement faithful gate replay.
