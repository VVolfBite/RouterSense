# RouterSense paper evaluation harness

本目录只记录冻结期论文评估框架的结构、合同和审计边界。

当前正式入口只有一个：

`python -m experiments.paper.cli <subcommand>`

当前子命令：

- `audit`
- `capture-trace`
- `build-traffic`
- `scheduling`
- `prediction`
- `hiding`
- `runtime-correctness`
- `aggregate`

约束：

- `experiments.paper` 只调用已有 `src.rs` public API。
- 不复制 planner / predictor / executor 实现。
- 缺能力时只标记 `MISSING_CAPABILITY` / `SEMANTIC_BLOCKER` / `SOFTWARE_BUG_CANDIDATE`，不在本轮顺手修正式代码。
