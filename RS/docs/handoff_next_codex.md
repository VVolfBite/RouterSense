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
- async_release CPU executable simulator
- transport-stress / EP replay offline 入口
- safe-U closure:
  - `RS_safe_barrier_criticality`
  - `RS_safe_gated_greedy`
  - `RS_safe_gated_maxweight`
  - `RS_safe_barrier_price`
- tensorized global dispatch-matrix gather for phase_sync prediction/prepared-plan bookkeeping
- runtime bridge candidates:
  - `routersense_joint_priority_phase_sync`
  - `routersense_joint_async_release_sim`
- offline FATE-style predictor:
  - `fate_style_history` (traffic baseline only)
  - `fate_style_linear` (traffic baseline only)
- expert prediction foundation:
  - `src/rs/runtime/online/megatron_ep/prediction/expert_trace.py`
  - `expert_to_traffic.py`
  - `expert_evaluation.py`
  - `gate_replay_predictor.py`
  - `traffic_calibration.py`

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
- 当前 online prepared-plan 应优先消费 `predicted_next_dispatch`；
  dispatch matrix 的全局构造方式必须是 tensorized gather，不能再走 Python object collective
- 当前 `fate_style_history` / `fate_style_linear` 只是 traffic-matrix baseline，不要再把它们当 faithful FATE predictor
- 当前真实 fixture 还没有 `expert_route_trace` / `source_expert_counts`；
  contribution 2 现在是 “expert foundation ready, GPU collection still required”
- 当前真实 fixture CPU 主线结论应先看：
  - `outputs/offline/m6h_safe_u_closure/replay_suite_summary.json`
  - `outputs/offline/m6k_cpu_closure/prediction_replay_summary.json`
  - `outputs/offline/m6k_cpu_closure/oracle_gap_summary.json`

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
  - runtime-lookahead paired B / raw U / safe U
  - execution-window paired B / raw U / safe U
  - prediction replay / oracle-predict
- prediction replay 现在会显式报告：
  - `expert_trace_available`
  - `expert_prediction_available`
  - `gate_replay_available`
  - `traffic_calibration_mode`
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
- `routersense_p0p1p2_hint` 是早期 runtime adapter，不再代表最终 RouterSense 主线
- 当前 offline 主线 safe-U：
  - `RS_safe_barrier_criticality`
  - `RS_safe_gated_greedy`
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
   - 当前 runtime-lookahead paired safe-U 结果
   - execution-window paired safe-U / raw U 结果
   - prediction replay 是否真能让 safe-U 受益
2. 基于 transport-stress / EP replay 评估 communication-only 空间，但不要把 stress 当主论文结论
3. 开 GPU 时优先验证：
   - `RS_safe_barrier_criticality` vs `birkhoff_phase_local`
   - prediction timing / layer interval / prepared-plan overlap
   - collect expert route trace:
     - `selected_experts`
     - `routing_weights`
     - `source_rank`
     - `layer_id`
     - `expert_to_rank_map`
   - dump:
     - `rank*_expert_route_trace.jsonl`
     - `rank*_source_expert_counts.jsonl`
     - `rank*_expert_to_traffic_audit.jsonl`
4. 如果 GPU 结果显示 phase_sync safe-U 仍弱，再推进 async_release executor integration

## 7. Repository rule

GitHub `main` 是外部审查 source of truth。

不要假设本地 deliverables 一定会存在。只要某个 contract、入口、handoff、evidence 脚本对审查重要，就必须提交并 push。
