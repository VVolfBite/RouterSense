> **Superseded Oracle note (2026-07-19):** formal `O_local`/`O_joint` now use the unified canonical bucket-wave exact model. Historical CT/atomic CP-SAT wording below is retained for chronology; use `docs/results/prediction_oracle_baseline_closure.md` for current numbers.

# Prediction / Oracle / Baseline Closure

- commit: `a3abe86457e772ff61de1ef82af0aac4eec964b4`
- cached: `False`
- selected_predictor: `history_linear_trend`

1. 联合调度相对 FIFO 平均、median、最好和最差提升多少？
   strongest joint heuristic `U_barrier_criticality_global_matching` vs FIFO: mean 1.93%, median 1.27%, best 8.34%, worst -2.42%.
2. 联合调度相对 Birkhoff 提升多少？
   `U_barrier_criticality_global_matching` vs `birkhoff_phase_local`: mean 2.05%, median 1.27%, best 8.34%, worst -2.42%.
3. `O_joint` 相对 `O_local` 平均改善多少？
   mean 6.49%, median 0.00%.
4. CT exact oracle 一共多少个 OPTIMAL 实例？
   32.
5. copy-current 相对 zero-hint 提升多少？
   mean 15.50%, median 17.57%.
6. copy-current 相对 perfect-trace hint 差多少？
   mean regret -1.07%; negative means copy-current beat perfect-trace hint under the current scheduler family on some windows.
7. copy-current 恢复了多少 perfect-hint 潜在收益？
   mean recovered ratio 121.84%, median 101.92%.
8. perfect-trace hint 是否稳定优于 zero-hint？
   no; it beat zero-hint on 85.94% of paired comparisons, with 9 windows/family-pairs showing no gain.
9. 如果不稳定，问题主要来自哪些 regime 或 scheduler family？
   failures concentrate in `ORACLE_ALSO_NO_GAIN` rows, mostly safe variants and a small set of late / weak-future windows; the closure did not observe a separate predictor-only failure regime where perfect-trace systematically hurt.
10. 各启发式距离 CT `O_joint` 还有多远？
    - birkhoff_phase_local_optimality_gap_to_O_joint: mean gap 9.43%, median 0.00%.
    - joint_copy_current_optimality_gap_to_O_joint: mean gap 4.48%, median 0.00%.
    - joint_perfect_trace_hint_optimality_gap_to_O_joint: mean gap 2.55%, median 0.00%.
    - joint_zero_hint_optimality_gap_to_O_joint: mean gap 3.85%, median 0.00%.
    - phase_barrier_fifo_optimality_gap_to_O_joint: mean gap 9.43%, median 0.00%.
    - safe_copy_current_optimality_gap_to_O_joint: mean gap 3.33%, median 0.00%.
    - safe_perfect_trace_hint_optimality_gap_to_O_joint: mean gap 1.41%, median 0.00%.
11. safe-U 的 median/p90/p99 CPU 开销？
    - gated_maxweight_matching: median 5290.8 us, p90 69506.7 us, p99 85293.1 us.
    - barrier_criticality_matching: median 5376.7 us, p90 66235.3 us, p99 78304.0 us.
12. safe-U 选择 U 和 B 的比例？
   gated_maxweight_matching: U=75.0%, B=25.0%; barrier_criticality_matching: U=93.8%, B=6.2%.
13. safe-U 在执行真值下是否真的避免退化？
   yes; projection-based safe selection avoided raw-U regression 100 times in total and produced 0 recorded wrong selections in this closure.
14. 预测准确度与调度收益是否相关？
   only weakly; Pearson(relative L1, regret)=0.0764245160590777, Spearman(relative L1, regret)=0.04192428559500989.
15. 哪些结论可以进入论文，哪些仍不能？
   can enter: exact small-instance O_joint<=O_local evidence, joint heuristics beating FIFO/Birkhoff on the replay fixture, and copy-current beating zero-hint on average. cannot yet enter as a universal claim: perfect-trace hint always improving the scheduler, or safe-U net end-to-end benefit on GPU.

## Pairing Audit

- pairing_missing_count: 0
- perfect_trace / actual_trace duplicate count: 0
- perfect_trace_hint kept in diagnostic comparisons: true
