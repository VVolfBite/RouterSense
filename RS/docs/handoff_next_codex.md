# RouterSense Handoff For Next Codex

## Current Commit

- `a803c43263ac575eab6572488db84d377657a3cd`
- branch: `main`

## Current Mainline Directories

### Online runtime

- `src/rs/runtime/online/megatron_ep/`
- primary entrypoints:
  - `host.py`
  - `lifecycle.py`
  - `control/plan_agreement.py`
  - `pending_window/adapter.py`

### Offline replay / trace

- `src/rs/runtime/offline/`
- `experiments/offline/`
- lightweight control replay parser:
  - `experiments/offline/replay_online_control_trace.py`

### Scheduling layer

- `src/rs/scheduling/`
- pure scheduling layer, no Megatron / executor dependency

### Experiment entrypoints

- `experiments/online/collect_native_ep_trace.py`
- `experiments/online/run_policy_correctness.py`
- `experiments/online/run_strategy_comparison.py`
- `experiments/offline/run_flow_schedule_study.py`

## Completed So Far

### 1. Execution audit hotfix

Completed:

- perf-mode scheduled plan now keeps minimal `task_ids`
- audit no longer false-fails on multi-payload P0 tasks
- task coverage checks are based on logical task ids
- birkhoff and routersense online correctness runs can pass audit

Relevant files:

- `src/rs/runtime/online/megatron_ep/execution/audit.py`
- `tests/contract/megatron_ep/test_execution_audit.py`

### 2. RouterSense fast path

Completed:

- `routersense_p0p1p2_hint` has a lighter online fast path
- still root-authoritative
- still wave-level
- still uses existing materialization and executor
- does not change payload layout or collective semantics

Important constraint:

- this is still online phase-local execution with prepared priority guidance
- it is not a full online multiphase live pending queue executor

### 3. Natural workload result

Current best natural 4GPU workload:

- `configs/comparison/tmp_comm_ramp_selected_4gpu.yaml`
- workload:
  - `configs/workload/comparison_256x128_prompts.json`

Observed conclusion:

- natural OLMoE 4GPU forward can reach about `5.5%` communication share
- this is usable as a conservative natural workload
- still compute-dominated overall

### 4. Online hot path contract

Added:

- `docs/runtime_online_hotpath_contract.md`

This document now defines:

- what may remain in perf hot path
- what must stay out of perf hot path
- fast path safety boundary
- online/offline analysis boundary

### 5. Control replay trace skeleton

Added:

- `docs/runtime_replay_trace_contract.md`
- `experiments/offline/replay_online_control_trace.py`
- replay trace writer path through:
  - `src/rs/runtime/online/megatron_ep/lifecycle.py`
  - `experiments/online/support/phase_executor_artifacts.py`

Current behavior:

- replay trace is default-off
- enabled by `observation.replay_trace_enabled=true`
- perf profile remains lightweight
- no tensor payload is saved in replay trace

## Current Important Paths

### Contracts / docs

- `docs/runtime_online_hotpath_contract.md`
- `docs/runtime_replay_trace_contract.md`
- `src/rs/runtime/online/megatron_ep/README.md`
- `src/rs/runtime/offline/README.md`
- `src/rs/scheduling/README.md`

### Replay trace implementation

- writer view builder:
  - `src/rs/runtime/online/megatron_ep/observation/views.py`
- runtime export hook:
  - `src/rs/runtime/online/megatron_ep/lifecycle.py`
- rank artifact flush:
  - `experiments/online/support/phase_executor_artifacts.py`
- offline parser:
  - `experiments/offline/replay_online_control_trace.py`
- tests:
  - `tests/contract/test_control_replay_trace.py`

## Explicit Do-Not-Do Items

Do not do these next by default:

- do not large-scale split `lifecycle.py`
- do not add heavy artifact dumps back into online perf hot path
- do not start full GPU benchmark sweeps
- do not keep blindly increasing natural batch sizes after `256x128`

## Recommended Next Order

### First

Use replay trace to analyze:

- all_gather scale
- broadcast scale
- wave count
- bucket count
- task ref count

### Second

Add a transport-stress / EP replay experiment line.

Reason:

- natural full forward remains compute-dominated
- communication-centric claims need a stronger communication-heavy setting

### Third

Fix global P2 matrix semantics.

Current known limitation:

- `p2_matrix_source` is still `replicated_local_row`
- not yet a real gathered global future matrix

### Fourth

Only after the above, continue RouterSense vs Birkhoff tuning.

## Current Config Guidance

Recommended current config for natural comparison:

- `configs/comparison/tmp_comm_ramp_selected_4gpu.yaml`

Bucket comparison only:

- `configs/comparison/tmp_comm_ramp_selected_bucket1024_4gpu.yaml`

Native-only reference:

- `configs/comparison/tmp_comm_ramp_256x128_disabled.yaml`

See also:

- `configs/comparison/README.md`

## Repository Status Rule

GitHub `main` is now the source of truth for external review.

Do not assume reviewers will read local deliverables.
Any new hot path contract, replay trace schema, experiment entrypoint, or config cleanup
must be pushed to the public repository before handoff.
