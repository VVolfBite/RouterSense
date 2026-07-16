# Paper evaluation configs

- `capability_audit.yaml`
  - 用途: 论文评估框架能力审计与 tiny smoke。
  - 环境: CPU 即可；如设置 `RS_MODEL_PATH`，trace capture 能力会被审计为可运行。
  - correctness: 只验证框架与正式 public API 是否接通。
  - timing eligible: 否。
  - runner: `python -m experiments.paper.cli audit`

- `trace_capture.yaml`
  - 用途: 单卡真实模型 trace capture 的正式配置入口。
  - 环境: GPU + 模型路径。
  - correctness: 是，采集真实 router trace。
  - timing eligible: 否。
  - runner: `python -m experiments.paper.cli capture-trace`

- `scheduling_value.yaml`
  - 用途: offline scheduling paired/oracle 评估入口。
  - 环境: CPU。
  - correctness: 是，比较 paired/local/joint/oracle。
  - timing eligible: 否。
  - runner: `python -m experiments.paper.cli scheduling`

- `prediction_value.yaml`
  - 用途: perfect/zero/shuffled 与正式 predicted 接口审计。
  - 环境: CPU。
  - correctness: 是。
  - timing eligible: 否。
  - runner: `python -m experiments.paper.cli prediction`

- `hiding_timeline.yaml`
  - 用途: timeline/hiding 能力审计入口。
  - 环境: CPU。
  - correctness: 部分，仅审计 current public API。
  - timing eligible: 否。
  - runner: `python -m experiments.paper.cli hiding`

- `runtime_correctness_gloo.yaml`
  - 用途: runtime correctness harness 入口。
  - 环境: CPU/Gloo；本轮只跑 single-process smoke。
  - correctness: 是，但不宣称 GPU/NCCL 性能。
  - timing eligible: 否。
  - runner: `python -m experiments.paper.cli runtime-correctness`
