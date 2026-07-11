# Prediction / Oracle / Baseline Closure

- commit: `adcd1d5784e9a4af26c98cb13498b667e3d2d917`
- cached: `False`
- selected_predictor: `history_linear_trend`
- CT OPTIMAL instance count: 32
- O_joint vs O_local mean improvement: 6.49%
- strongest joint baseline: `U_gated_maxweight_matching`
- strongest joint vs FIFO median gain: 1.24%
- strongest joint vs Birkhoff median gain: 1.31%
- copy-current vs zero mean gain: 15.50%
- copy-current vs perfect-trace-hint mean regret: -1.07%
- recovered perfect-hint gain mean: 121.84%
- perfect-trace-hint better-than-zero rate: 85.94%
- safe-U median CPU overhead (primary family): 7746.6 us
- safe-U select U/B ratio (primary family): 0.750 / 0.250

## Heuristic Gaps to O_joint

- birkhoff_phase_local_optimality_gap_to_O_joint: mean gap 9.43%
- joint_copy_current_optimality_gap_to_O_joint: mean gap 4.48%
- joint_perfect_trace_hint_optimality_gap_to_O_joint: mean gap 2.55%
- joint_zero_hint_optimality_gap_to_O_joint: mean gap 3.85%
- phase_barrier_fifo_optimality_gap_to_O_joint: mean gap 9.43%

## Predictor Correlation

- pearson(relative_l1, regret): 0.058955650809424644
- spearman(relative_l1, regret): 0.044418609139054654
- pearson(cosine, regret): 0.10234936231951118
- spearman(cosine, regret): 0.11808346129807817

## Pairing Audit

- pairing_missing_count: 0
- perfect_trace / actual_trace duplicate count: 0
- perfect_trace_hint kept in diagnostic comparisons: true
