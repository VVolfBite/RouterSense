# RouterSense Handoff For Next Codex

## Current Mainline

- branch: `main`
- current commit: always check with `git rev-parse HEAD`

这个 handoff 只描述当前 mainline，不再保留旧 distributed-EP bringup 叙事。

## 1. Online runtime mainline

当前真实 online 主线：

- `src/rs/runtime/online/megatron_ep/`
- `experiments/online/run_strategy_comparison.py`

已经完成：

- execution audit hotfix
- perf artifact slimming
- routersense fast path
- public runtime surface：
  - `runtime.line=phase_sync`
  - `runtime.line=async_release`
  - `runtime.output_mode=paper`
  - `runtime.output_mode=debug_replay`
- control replay trace skeleton
- natural 4GPU `256x128` workload主线
- async_release shadow-only skeleton
- transport-stress / EP replay offline 入口
- prepared-plan global P2 matrix gather for phase_sync when EP group is available

当前明确不要做：

- 不要把重 debug artifact 塞回 perf hot path
- 不要大拆 `lifecycle.py`
- 不要绕过 root agreement
- 不要把 fast path 写成本地 greedy
- 不要把 `async_release` 偷偷 fallback 到 `phase_sync`

重要解释：

- 当前 online RouterSense 仍然是 prediction-aware phase-local runtime policy
- 它不是 full online multiphase live pending queue executor
- `async_release` 当前只有 shadow-only skeleton，还没有 executor integration
- 当前 prepared-plan 的 `p2_matrix_source` 在真实分布式 phase_sync 下应优先是 `gathered_global_matrix`；无分布式环境才 fallback 到 `replicated_local_row`

## 2. Offline / replay mainline

当前 offline 分析主线：

- `src/rs/runtime/offline/`
- `experiments/offline/replay_online_control_trace.py`
- `experiments/offline/build_replay_fixture_from_control_trace.py`
- `experiments/offline/run_real_trace_evidence_suite.py`
- `experiments/offline/run_transport_stress_replay.py`

当前能力：

- 读取 lightweight control replay trace
- 汇总 control-plane object scale
- 从 replay trace 构建 offline fixture + audit summary
- 跑三类证据表：
  - phase-sync-compatible
  - execution-window joint upper bound
  - prediction / oracle-predict
- 把真实 fixture 进一步压成 communication-only transport-stress replay

当前限制：

- 不替代真实 GPU benchmark
- 不精确模拟 NCCL 等待
- 不应把 execution-window upper bound 表述成当前 online 已实现收益

## 3. Scheduling mainline

当前 runtime-facing 策略主线：

- `src/rs/scheduling/`

解释：

- `birkhoff_phase_local` 是当前 online 可执行的强 phase-local baseline
- `routersense_p0p1p2_hint` 是当前 prediction-aware runtime policy
- `B_birkhoff_wave` / `U_*` 属于 offline joint scheduling evidence，不应塞回 online perf hot path

## 4. 当前推荐配置与入口

Natural workload mainline：

- `configs/comparison/natural_256x128_4gpu.yaml`
- workload:
  - `configs/workload/comparison_256x128_prompts.json`

Public runtime 说明：

- `docs/runtime_public_entrypoints.md`

当前代码结构索引：

- `docs/current_code_structure_index.md`

论文证据链：

- `docs/paper_evidence_chain.md`

## 5. 关键实现文件

Online runtime：

- `src/rs/runtime/online/megatron_ep/host.py`
- `src/rs/runtime/online/megatron_ep/lifecycle.py`
- `src/rs/runtime/online/megatron_ep/control/plan_agreement.py`
- `src/rs/runtime/online/megatron_ep/pending_window/adapter.py`
- `src/rs/runtime/online/megatron_ep/pending_window/policy_adapter.py`
- `src/rs/runtime/online/megatron_ep/observation/views.py`
- `src/rs/runtime/online/megatron_ep/async_release/`

Offline / replay：

- `experiments/offline/replay_online_control_trace.py`
- `experiments/offline/build_replay_fixture_from_control_trace.py`
- `experiments/offline/run_replay_fixture_policy_suite.py`
- `experiments/offline/run_real_trace_evidence_suite.py`
- `experiments/offline/run_transport_stress_replay.py`

## 6. 推荐下一步顺序

1. 基于 real trace evidence suite 先区分：
   - 当前 phase_sync-compatible 结果
   - execution-window joint upper bound
   - prediction / oracle-predict 空间
2. 基于 transport-stress / EP replay 评估 communication-only 空间
3. 接真实 online predictor，而不是只靠 gathered previous-layer matrix
4. 再决定是否推进 async_release executor integration

## 7. Repository rule

GitHub `main` 是外部审查 source of truth。

不要假设本地 deliverables 一定会存在。只要某个 contract、入口、handoff、evidence 脚本对审查重要，就必须提交并 push。
