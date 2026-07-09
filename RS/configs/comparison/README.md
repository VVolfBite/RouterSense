# Comparison Configs

这个目录现在分两类：

## 1. 当前推荐的自然场景配置

围绕 4GPU OLMoE natural forward 的最新结论，当前保留并推荐优先使用：

- `tmp_comm_ramp_256x128_disabled.yaml`
  - 只跑 native/disabled
  - 用于确认自然场景下的通信占比
  - 当前已知可稳定跑通，通信占比约 5.5%

- `tmp_comm_ramp_selected_4gpu.yaml`
  - 当前主线三策略对比
  - workload 固定为 `comparison_256x128_prompts.json`
  - `bucket_rows=0`

- `tmp_comm_ramp_selected_bucket1024_4gpu.yaml`
  - 与上面同 workload
  - 仅用于 bucket 粒度对照

如果只是继续当前论文主线实验，优先用这三份。

## 2. paper / legacy 保留配置

以下文件暂时保留，不代表它们都是当前默认入口：

- `paper_core_*`
- `paper_phase_local_*`
- `wire_slim_core_4gpu_64.yaml`
- `default.yaml`

这些主要用于：

- 历史对比
- 论文不同表格的固定 workload
- 更早的 phase-local / wire-size 分析

## 清理原则

已经删除上一轮纯筛 workload 用的临时 ramp 配置，例如：

- 32x128 / 64x128 / 96x128 / 128x128 disabled ramp
- 128x64 / 256x64 / 512x32 / 512x64 / 384x128 / 512x128 探测配置
- 8x16 trimmed smoke config

原因很简单：

- 这些配置已经完成它们的探索用途
- 当前主线结果收敛到 `256x128`
- 不希望后续继续误用过时 tmp 配置
