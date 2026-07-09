# Paper Evidence Chain

这个文档把 RouterSense 论文的三条核心贡献和当前代码入口对齐。

## Claim 1

**Multi-phase joint scheduling has real optimization space**

对应代码：

- `experiments/offline/build_replay_fixture_from_control_trace.py`
- `experiments/offline/run_real_trace_evidence_suite.py`
- `experiments/offline/run_transport_stress_replay.py`
- `src/rs/scheduling/`
  - `B_birkhoff_wave`
  - `U_gated_maxweight_matching`
  - `U_barrier_criticality_global_matching`

当前状态：

- 已能把真实 online trace 转成 replay fixture。
- 已能在 offline execution-window 语义下比较 `B_birkhoff_wave` 与 `U_*`。
- 已能把真实 fixture 进一步压成 communication-only transport-stress replay 报告。
- 这条证据链当前是 offline 可验证，不等于 online 已实现。

## Claim 2

**Cross-layer prediction can improve scheduling and reduce online cost**

对应代码：

- `experiments/offline/run_real_trace_evidence_suite.py`
- `experiments/offline/replay_fixture_policy_study.py`
- `src/rs/runtime/online/megatron_ep/pending_window/`
- `src/rs/runtime/online/megatron_ep/async_release/`

当前状态：

- offline 端已能区分：
  - `zero_hint`
  - `copy_current_dispatch`
  - `perfect_trace`
  - `actual_trace`
- online 侧现在已经有 tensorized dispatch-matrix gather 和 lightweight predictor contract：
  - `ZeroHintPredictor`
  - `CopyCurrentDispatchPredictor`
- prepared-plan 应消费 `predicted_next_dispatch`，而不是把 gathered current matrix 本身写成 predictor。
- online 端当前只有 phase_sync 下的 prediction-aware policy。
- 但这还不等于真实 next-layer predictor：当前只有 zero/copy-current 级别的轻量 predictor，真实 predictor 仍未接入。

## Claim 3

**RouterSense is a real reproducible online runtime**

对应代码：

- `src/rs/runtime/online/megatron_ep/`
- `experiments/online/run_strategy_comparison.py`
- `experiments/online/support/runtime_presets.py`
- `experiments/offline/replay_online_control_trace.py`

当前状态：

- `phase_sync` 是当前真实可执行、可审计、可 replay 的 online 主线。
- replay trace、execution audit 和 public runtime surface 已经打通。
- `async_release` 当前只有 shadow-only skeleton，还没有 executor integration。

因此当前论文口径必须保持诚实：

- online reproducible runtime 已成立；
- full async joint execution 仍是下一阶段工作。
