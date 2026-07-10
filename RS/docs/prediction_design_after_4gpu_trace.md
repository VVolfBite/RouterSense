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
- With `phase_context` P0 actual matrices and `hidden_only=4096`, the expert-count-to-traffic mapping can be aligned.

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

## Recommended Sequence

1. Fix and lock the expert-to-traffic semantic evaluation path.
2. Add or expose debug-only policy decision explanations for P2 score/order effects.
3. Repair P2 consumption in `RS_safe_barrier_criticality` and `RS_safe_gated_greedy`.
4. Compare traffic predictors against expert predictors.
5. Only then implement faithful gate replay.
