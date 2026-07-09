# RouterSense Handoff For Next Codex

## Current Mainline

- branch: `main`
- current commit should always be checked with `git rev-parse HEAD`

This handoff intentionally describes only the current mainline.
Older distributed-EP bringup phases are no longer the active source of truth here.

## 1. Online runtime mainline

Current real runtime mainline:

- `src/rs/runtime/online/megatron_ep/`
- `experiments/online/run_strategy_comparison.py`

Already completed on this line:

- execution audit hotfix
- perf artifact slimming
- routersense fast path
- natural 4GPU `256x128` workload result
- control replay trace skeleton
- public runtime entry narrowing:
  - `runtime.line`
  - `runtime.output_mode`

Current constraints:

- do not put heavy debug artifact back into perf hot path
- do not large-scale split `lifecycle.py`
- do not bypass root agreement
- do not turn fast path into local greedy

Important interpretation:

- current online RouterSense is still a prediction-aware phase-local runtime policy
- it is not a full online multiphase live pending queue executor
- current public runtime lines:
  - `phase_sync` (implemented)
  - `async_release` (declared, not implemented)
- current public output modes:
  - `paper`
  - `debug_replay`

## 2. Offline / replay mainline

Current offline analysis mainline:

- `src/rs/runtime/offline/`
- `experiments/offline/replay_online_control_trace.py`

What it does now:

- reads lightweight control replay traces
- summarizes control-plane object scale
- reports wave / bucket / task-ref / wire-size statistics

What it does not do:

- it does not replace real GPU benchmark
- it does not simulate NCCL waiting precisely
- it does not perform full strategy re-planning yet

## 3. Scheduling mainline

Current runtime-facing strategy mainline:

- `src/rs/scheduling/`

This is the real policy contract entry used by current online/offline code.

Interpretation:

- `birkhoff_phase_local` is the current online-executable strong phase-local baseline
- `routersense_p0p1p2_hint` is the current prediction-aware runtime policy
- oracle / heavy joint schedulers belong to offline or theoretical upper-bound analysis
- they should not be moved into online perf hot path

## 4. Current recommended config path

Natural workload mainline:

- `configs/comparison/natural_256x128_4gpu.yaml`
- workload:
  - `configs/workload/comparison_256x128_prompts.json`

Legacy references still kept:

- `configs/comparison/tmp_comm_ramp_256x128_disabled.yaml`
- `configs/comparison/tmp_comm_ramp_selected_bucket1024_4gpu.yaml`

See:

- `configs/comparison/README.md`
- `docs/runtime_public_entrypoints.md`

## 5. Key documents

- `docs/runtime_online_hotpath_contract.md`
- `docs/runtime_replay_trace_contract.md`
- `docs/current_code_structure_index.md`

## 6. Key implementation files

Online runtime:

- `src/rs/runtime/online/megatron_ep/host.py`
- `src/rs/runtime/online/megatron_ep/lifecycle.py`
- `src/rs/runtime/online/megatron_ep/control/plan_agreement.py`
- `src/rs/runtime/online/megatron_ep/pending_window/adapter.py`
- `src/rs/runtime/online/megatron_ep/pending_window/policy_adapter.py`
- `src/rs/runtime/online/megatron_ep/observation/views.py`

Replay trace:

- writer path:
  - `src/rs/runtime/online/megatron_ep/lifecycle.py`
  - `experiments/online/support/phase_executor_artifacts.py`
- parser:
  - `experiments/offline/replay_online_control_trace.py`
  - `experiments/offline/build_replay_fixture_from_control_trace.py`
- tests:
  - `tests/contract/test_control_replay_trace.py`

## 7. Explicit do-not-do list

Do not do these by default next:

- do not large-scale split `lifecycle.py`
- do not re-inflate perf hot path artifact volume
- do not run full GPU benchmark sweeps first
- do not keep blindly increasing natural batch sizes beyond the current 256x128 line
- do not re-expose low-level runtime knobs in recommended configs
- do not silently map `async_release` back to `phase_sync`

## 8. Recommended next order

1. Analyze replay trace:
   - all_gather scale
   - broadcast scale
   - wave count
   - bucket count
   - task-ref count
2. Bridge real rank traces into offline scheduling fixtures
3. Add transport-stress / EP replay
4. Fix global P2 matrix
5. Then continue RouterSense vs Birkhoff tuning

## 9. Repository rule

GitHub `main` is the external review source of truth.

Do not assume local deliverables will be available.
If a new contract, replay schema, config cleanup, or handoff note matters for review,
it must be committed and pushed.
