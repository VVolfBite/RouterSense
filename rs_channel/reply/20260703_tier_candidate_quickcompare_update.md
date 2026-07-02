本轮已完成并准备推送：

- 候选体系从“淘汰清单”改为“三档候选清单”
  - `Tier 1`: 定论候选池
  - `Tier 2`: 优化晋级池
  - `Tier 3`: 快速淘汰池
- 第二档实验规格已改为：
  - `N=8`
  - `sample=64`
- 统一保留单一标准非重复语料：
  - `RS/artifacts/poc_line1/prompt_sets/olmoe_oasst256_unique.jsonl`
- 新增实验预设配置：
  - `RS/experiments/poc_line1/configs/candidate_tiers.json`
- `exp_pairwise_candidate_compare.py` 支持：
  - `--config-json`
  - `--config-key`
- `build_prompt_mix.py` 支持：
  - 总量限制
  - 文本字段选择
  - 条件字段过滤

本轮 quick compare 的核心结论：

- `wave` 不会颠覆头部格局，`U` 仍然第一梯队
- 最能从 `wave` 获益的是 `B`
- 在 `N=8` 小样本下，`U_wave` 已接近当前 `fluid wave oracle` 参考上限

未随本次提交进入 git：

- `RS/artifacts/poc_line1/full_sequence_trace_olmoe_mix200_unique_v1`
- `RS/artifacts/poc_line1/quick_wave_atomic_n8_s64`
- 所有 smoke / quick 运行目录
