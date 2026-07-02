本轮已完成并待推送：

- `multiphase_global.py`
  - 新增 `atomic` 执行模式
  - 新增 4 个 atomic U_ 调度器
  - audit `wave_count` 改为优先读取 `wave_id`
- `analysis.py`
  - 默认主报告名切到 `U_`/`B_`
  - 纳入 atomic U_ 候选
- `strategies.py` / `scheduler/__init__.py` / `evaluation/__init__.py`
  - 注册并导出 atomic U_ 调度器
  - 保留旧 `O_` alias 兼容历史脚本
- 新增实验：
  - `experiments/poc_line1/exp_ablation_fluid_vs_joint.py`
  - `experiments/poc_line1/build_prompt_mix.py`
- 测试：
  - `pytest -q tests/test_poc_line1.py -q` 通过

当前后台仍在跑：

- OLMoE 200 条独立 prompt trace 重采
  - 输出目录：`RS/artifacts/poc_line1/full_sequence_trace_olmoe_mix200_unique_v1`

已完成 smoke：

- `smoke_ablation_atomic`
- `smoke_n32_atomic`

关键 smoke 结论：

- `N=32` 下 atomic 版显著弱于 fluid 版
- 联合收益不只来自 barrier/joint release，本质上仍高度依赖 fluid re-matching
