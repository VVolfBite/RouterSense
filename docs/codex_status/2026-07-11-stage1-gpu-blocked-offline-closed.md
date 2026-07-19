# 2026-07-11 Stage1 GPU Blocked, Offline Closed

## Summary

- Offline stage1 closure has been rerun and formalized at `outputs/offline/stage1_paper_closure_final`.
- 4GPU `B2` progressed far enough to show real nonzero async `P0` payload execution.
- `B2` still fails before `P1` because the stored runtime joint window state reaches `before_token_combine` with zero `actual_p0_full_row_matrix` / `inferred_p1_row_matrix`.
- `C2` and `A2` were not advanced because the execution discipline requires `B2` to pass first.

## Evidence

- Offline final summary: `outputs/offline/stage1_paper_closure_final/final_offline_summary.json`
- Offline paper tables: `outputs/offline/stage1_paper_closure_final/paper_ready_tables.md`
- GPU blocker summary: `outputs/distributed/stage1_gpu_blocker_summary.json`
- Failing B2 artifact: `outputs/distributed/run_b2_20260711_144200/candidate/b2_candidate/failure_report.json`
- Real nonzero P0 async evidence: `outputs/distributed/run_b2_20260711_145000/candidate/heartbeat-rank0.jsonl`

## Main conclusions

- Strongest offline baseline remains `phase_barrier_fifo`.
- Shared-core proxy still shows positive joint-only gain:
  - `gated_greedy`: `14.91%`
  - `barrier_criticality_matching`: `11.15%`
- Exact small oracle still shows `13.33%` joint-vs-local improvement.
- Predictor selection remains `zero_hint` on held-out schedule regret.
- Oracle traffic does not improve held-out schedule regret under the current scheduler core summary.
- Real 4GPU runtime async path is reachable, but full `P1` lifecycle correctness is not yet proven.

## Progress table

- 联合调度机会: `80% -> 90%`
  - evidence: shared-core proxy gains and oracle gap tables
  - remaining: validate on real 4GPU `A2`
- 离线完整基线与 oracle: `65% -> 90%`
  - evidence: `stage1_paper_closure_final`
  - remaining: expand exact oracle sample count beyond the current 6-row formal set if needed for final paper
- 流量生成与预测数据链: `80% -> 88%`
  - evidence: reproducible fixture manifest and unified replay closure
  - remaining: finish online `B2` lifecycle pass
- 非 oracle 预测器: `55% -> 72%`
  - evidence: held-out predictor selection and schedule-regret tables
  - remaining: no positive online-eligible predictor beyond `zero_hint`
- 预测带来的调度收益: `35% -> 45%`
  - evidence: held-out regret and oracle-vs-zero analysis
  - remaining: current evidence does not support a positive claim
- Async P2P 执行器: `75% -> 84%`
  - evidence: real nonzero `P0` wave execution on 4GPU
  - remaining: `P1` lifecycle/materialization bug
- Runtime 联合规划链: `70% -> 82%`
  - evidence: real host path reaches joint async `P0`
  - remaining: stored `P1` plan state must match real combine-time layout
