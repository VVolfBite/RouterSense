# 2026-07-11 Stage1 Parallel Closure

## Mainline Progress

- 联合调度机会: 80% -> 90%
  evidence: outputs/offline/stage1_paper_closure/shared_core_b_u.csv, outputs/offline/stage1_paper_closure/host_projection_gap.csv
  remaining_work: 4GPU 上验证 runtime joint-only gain 是否能兑现到真实 transport
  gpu_dependency: True
- 离线完整基线与 oracle: 65% -> 90%
  evidence: outputs/offline/stage1_paper_closure/baseline_summary.csv, outputs/offline/stage1_paper_closure/oracle_gap.csv, outputs/offline/stage1_paper_closure/paper_ready_tables.md
  remaining_work: 仅剩换更多模型/trace 扩表，不再缺统一 runner
  gpu_dependency: False
- 流量生成与预测数据链: 80% -> 90%
  evidence: outputs/distributed/run_stage1_runtime_integrated_gloo_gate/summary.json, outputs/offline/stage1_paper_closure/predictor_validation.csv
  remaining_work: 4GPU B2 上收集真实 prediction lifecycle artifact
  gpu_dependency: True
- 非 oracle 预测器: 55% -> 68%
  evidence: outputs/offline/stage1_paper_closure/predictor_validation.csv, outputs/offline/stage1_paper_closure/predictor_selection.json
  remaining_work: 当前 held-out regret 最优仍是 zero_hint，后续需要真正提高 online-eligible predictor
  gpu_dependency: False
- 预测带来的调度收益: 35% -> 40%
  evidence: outputs/offline/stage1_paper_closure/prediction_schedule_regret.csv, outputs/offline/stage1_paper_closure/prediction_failure_taxonomy.csv
  remaining_work: 当前统一离线结论仍是 prediction mostly neutral；需要 4GPU A2 验证 runtime consumption
  gpu_dependency: True
- Async P2P 执行器: 80% -> 88%
  evidence: outputs/distributed/run_stage1_gloo_e2e_gate/summary.json, outputs/distributed/run_stage1_runtime_integrated_gloo_gate/summary.json
  remaining_work: 只差 4GPU NCCL C2 correctness 和 A2 performance
  gpu_dependency: True
- Runtime 联合规划链: 72% -> 88%
  evidence: outputs/distributed/run_stage1_runtime_integrated_gloo_gate/summary.json, outputs/pre_gpu_final_code_audit.json
  remaining_work: 只差 4GPU B2/C2/A2 实跑和一次针对真实瓶颈的调优
  gpu_dependency: True

## Key Answers

- strongest baseline: `phase_barrier_fifo`
- strict common-core B/U completed: `False`
- selected predictor: `zero_hint`
- held-out prediction error: `0.75`
- held-out schedule regret: `0.0`
- oracle prediction schedule gain vs zero: `None`
- selected predictor gain vs zero: `0.0`
- prediction failure taxonomy: `{'PREDICTION_NEUTRAL': 48}`
- runtime integrated Gloo passed: `True`
- low-level Gloo passed: `True`
- B2/C2/A2 GPU bodies implemented: `True`
