本轮新增并已准备推送：

- `B_wave` 公平性补齐
  - `B_birkhoff_wave`
  - `B_barrier_aware_birkhoff_wave`
  - 语义：保持 phase-local 基线排序信号，但允许 barrier release 后做 wave 级 interleave

- `U/B` 四象限公平性实验补齐
  - `B_atomic / B_wave / U_atomic / U_wave`
  - 实验入口：
    - `RS/experiments/poc_line1/exp_ablation_fluid_vs_joint.py`

- `wave oracle` 两条新增线
  - `pairwise_wave_oracle`
    - atomic/chunk 语义
    - 用于主实验里的强参考
  - `pairwise_fluid_wave_oracle`
    - fluid/wave 语义
    - 用于小样本理论上限分析

- oracle 元数据补全
  - `solve_time_ms`
  - `objective`
  - `best_bound`
  - `optimality_gap`
  - `time_limit_ms`

- 停止规则
  - 预算：`500ms`
  - 连续 `3` 次相对改善 `< 3%` 提前停

- 测试
  - `cd RS && pytest -q tests/test_poc_line1.py -q` 通过

当前仍在后台运行，未随本次提交进入 git：

- `RS/artifacts/poc_line1/full_sequence_trace_olmoe_mix200_unique_v1`
- `RS/artifacts/poc_line1/olmoe_mix200_n32_final_compare`
- `RS/artifacts/poc_line1/olmoe_mix200_n32_ablation`

当前关键机制性结论：

- `B_wave` 会显著变强，因此用 `B_atomic` 直接对比 `U_wave` 不公平
- `U_atomic` 相对 `B_wave` 的纯联合调度增益不大
- `U_wave` 的主要额外收益仍来自 wave/fluid 分流能力
