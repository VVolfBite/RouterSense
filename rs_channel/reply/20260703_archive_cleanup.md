清理和重组已完成。

现在受 git 管控的归档根目录是：
`/root/autodl-tmp/RouterSense/archive/backup/README.md`

当前只保留 3 个存档目录，目录名都采用“时间戳 + 实验内容”：

1. `20260703_cross_layer_prediction_validity`
   - README:
     `/root/autodl-tmp/RouterSense/archive/backup/20260703_cross_layer_prediction_validity/README.md`
   - docs:
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_cross_layer_prediction_validity/docs/poc_line1_status.md`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_cross_layer_prediction_validity/docs/router_trace_schema.md`
   - artifact:
     `/root/autodl-tmp/RouterSense/archive/backup/20260703_cross_layer_prediction_validity/artifacts/cross_layer_report_batch500_rr_v2`
   - report:
     `/root/autodl-tmp/RouterSense/archive/backup/20260703_cross_layer_prediction_validity/reports/20260701_poc_line1_gate_closure.md`

2. `20260703_oracle_fast_gap_study`
   - README:
     `/root/autodl-tmp/RouterSense/archive/backup/20260703_oracle_fast_gap_study/README.md`
   - docs:
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_oracle_fast_gap_study/docs/poc_line1_status.md`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_oracle_fast_gap_study/docs/multiphase_global_matching_study.md`
   - artifacts:
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_oracle_fast_gap_study/artifacts/pairwise_fast_report_batch500_rr_v3`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_oracle_fast_gap_study/artifacts/pairwise_oracle_report_batch500_v3`
   - report:
     `/root/autodl-tmp/RouterSense/archive/backup/20260703_oracle_fast_gap_study/reports/20260702_fast_scheduler_batch500_result.md`

3. `20260703_execution_window_multiscale`
   - README:
     `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/README.md`
   - docs:
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/docs/multiphase_global_matching_study.md`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/docs/architecture_boundary.md`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/docs/phase0c_distributed_ep_contract.md`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/docs/future_ep_weight_and_communication_contract.md`
   - artifacts:
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/artifacts/execution_window_n8_sample64_bo`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/artifacts/execution_window_n16_sample64_bo`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/artifacts/execution_window_n30_sample64_bo`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/artifacts/execution_window_n8_sample16_oracle_focus`
   - reports:
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/reports/20260703_execution_window_n8_n16_n30_sample64.md`
     - `/root/autodl-tmp/RouterSense/archive/backup/20260703_execution_window_multiscale/reports/20260703_execution_window_n8_oracle_focus.md`

之前顶层 `archive/backup/poc_line1` 的旧备份已经清掉；目前顶层 `archive/backup/` 里只有这 3 个核心存档。

另外，新建了总文档目录：
- `/root/autodl-tmp/RouterSense/archive/backup/docs/ELIMINATION_LOG.md`
- `/root/autodl-tmp/RouterSense/archive/backup/docs/new-direction.md`
- `/root/autodl-tmp/RouterSense/archive/backup/docs/poc-line-1.md`
- `/root/autodl-tmp/RouterSense/archive/backup/docs/poc1-legacy-task.md`
- `/root/autodl-tmp/RouterSense/archive/backup/docs/poc1-refined.md`
- `/root/autodl-tmp/RouterSense/archive/backup/docs/rs-task.md`

其中无后缀的根目录历史文本文件已转存为 `.md` 副本，原文件保留不动。

另外，`RS/artifacts/poc_line1/` 现在只保留 trace 输入，活动输出树已清空中间筛选、盲测、旧 smoke 和过时 duplex 结果。

关于 `legacy/`：
- 主线 `RS/src`、`RS/experiments`、`RS/deploy` 没有运行时依赖它。
- 测试里还有显式约束，禁止主线代码 import `legacy`。
- 它目前只保留历史参考价值：旧 POC、文档、数据样例、以及 full-package 打包场景。
