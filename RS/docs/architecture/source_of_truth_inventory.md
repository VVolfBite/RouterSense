## Source Of Truth Inventory

This inventory reflects the repository state after the pre-evaluation mainline consolidation pass.
It is based on the current import graph and test-covered entrypoints, not on directory names alone.

### 1. Formal online Megatron EP runtime

Canonical implementation lives under `src/rs/runtime/online/megatron_ep/`.

Primary implementation files:

- `src/rs/runtime/online/megatron_ep/_host_impl.py`
- `src/rs/runtime/online/megatron_ep/_lifecycle.py`
- `src/rs/runtime/online/megatron_ep/_observation.py`
- `src/rs/runtime/online/megatron_ep/contracts.py`
- `src/rs/runtime/online/megatron_ep/observer.py`
- `src/rs/runtime/online/megatron_ep/trace_writer.py`
- `src/rs/runtime/online/megatron_ep/p2_provider.py`
- `src/rs/runtime/online/megatron_ep/p2_contracts.py`
- `src/rs/runtime/online/megatron_ep/phase/context_builder.py`
- `src/rs/runtime/online/megatron_ep/phase/layout_join.py`
- `src/rs/runtime/online/megatron_ep/phase/validation.py`
- `src/rs/runtime/online/megatron_ep/control/contracts.py`
- `src/rs/runtime/online/megatron_ep/control/_plan_agreement_impl.py`
- `src/rs/runtime/online/megatron_ep/control/mailbox.py`
- `src/rs/runtime/online/megatron_ep/control/state_machine.py`
- `src/rs/runtime/online/megatron_ep/control/timeline.py`
- `src/rs/runtime/online/megatron_ep/control/validation.py`
- `src/rs/runtime/online/megatron_ep/execution/transport_adapter.py`
- `src/rs/runtime/online/megatron_ep/execution/sync_wave_executor.py`

Public re-export entrypoints:

- `src/rs/runtime/online/megatron_ep/host.py`
- `src/rs/runtime/online/megatron_ep/runtime.py`
- `src/rs/runtime/online/megatron_ep/lifecycle.py`
- `src/rs/runtime/online/megatron_ep/artifact_recorder.py`
- `src/rs/runtime/online/megatron_ep/policy_adapter.py`

Current status:

- Canonical runtime logic is in `src/rs/runtime/online/megatron_ep`.
- `integrations/megatron_ep/*` still exists as compatibility / experiment surface.
- `src/rs/runtime/online/megatron_ep` still depends on `rs.scheduling.policy.agreement`, which is a remaining layering defect.

### 2. Offline POC-line1 trace / matrix / predictor / solver

Current formalized locations:

- Trace:
  - `src/rs/runtime/offline/trace/olmoe.py`
  - `src/rs/runtime/offline/trace/qwen.py`
  - `src/rs/runtime/offline/trace/schema.py`
  - `src/rs/runtime/offline/trace/loader.py`
- Traffic / matrix:
  - `src/rs/runtime/offline/traffic/matrix_builder.py`
  - `src/rs/runtime/offline/traffic/placement.py`
  - `src/rs/runtime/offline/traffic/flow_window.py`
- Prediction:
  - `src/rs/runtime/offline/prediction/cross_layer.py`
  - `src/rs/runtime/offline/prediction/calibration.py`
  - `src/rs/runtime/offline/prediction/artifact.py`
- Scheduling:
  - `src/rs/scheduling/contracts.py`
  - `src/rs/scheduling/matching.py`
  - `src/rs/scheduling/validation.py`
  - `src/rs/scheduling/phase_local/*`
  - `src/rs/scheduling/baselines/*`
  - `src/rs/scheduling/reference/*`
  - `src/rs/scheduling/multiphase/global_ready_set_impl.py`

Current status:

- The formal offline tree exists.
- Several `experiments/offline/*` entrypoints still delegate to `experiments/poc_line1/*`.
- Old `src/rs/scheduler`, `src/rs/evaluation`, and `src/rs/trace` are still present and still part of the compatibility surface.

### 3. `src/rs` wrappers vs real implementation

Known wrapper / shim style files:

- `src/rs/runtime/online/megatron_ep/host.py`
- `src/rs/runtime/online/megatron_ep/runtime.py`
- `src/rs/runtime/online/megatron_ep/lifecycle.py`
- `src/rs/runtime/online/megatron_ep/phase/contracts.py`
- `src/rs/runtime/online/megatron_ep/execution/bucketizer.py`
- `src/rs/runtime/online/megatron_ep/execution/layout_validation.py`
- `src/rs/runtime/online/megatron_ep/phase/_layout_join_impl.py`
- `src/rs/scheduler/global_matching.py`
- `src/rs/scheduling/phase_local/fifo.py`
- `src/rs/scheduling/phase_local/aurora_fixed.py`
- `src/rs/scheduling/phase_local/fast_bvn_fixed.py`
- `src/rs/scheduling/multiphase/global_ready_set.py`
- `src/rs/scheduling/reference/oracle_guided.py`
- `src/rs/scheduling/baselines/birkhoff.py`
- `src/rs/scheduling/baselines/greedy.py`

These wrappers are acceptable only when they are one-way shims into the formal implementation and contain no canonical logic.

### 4. `integrations/` files still containing canonical logic

These still contain meaningful logic and are not yet reduced to pure deprecated shims:

- `integrations/megatron_ep/exp_phase_executor.py`
- `integrations/megatron_ep/exp_injection_smoke.py`
- `integrations/megatron_ep/collect_native_ep_trace.py`
- `integrations/megatron_ep/verify_env.py`
- `integrations/megatron_ep/probe_dispatch_boundary.py`
- `integrations/megatron_ep/probe_p1_pretransport.py`
- `integrations/megatron_ep/build_watchdog_report.py`
- `integrations/megatron_ep/build_policy_injection_proof.py`
- `integrations/megatron_ep/compare_phase_executor_outputs.py`
- `integrations/megatron_ep/compare_noop_equivalence.py`
- `integrations/megatron_ep/smoke_native_ep.py`

Compatibility shims already in place:

- `integrations/megatron_ep/native_runtime.py`
- `integrations/megatron_ep/routersense/dispatcher_facade.py`
- `integrations/megatron_ep/routersense/execution/fifo_policy.py`

### 5. Old paths still imported by formal or quasi-formal paths

Remaining old namespace usage of concern:

- `experiments/online/bench_native_ep.py` imports `rs.online.olmoe_ep`
- `experiments/online/collect_native_ep_trace.py` imports `rs.online.olmoe_ep`
- `experiments/distributed/*` imports `rs.runtime.distributed_ep`
- `tests/test_online_*`, `tests/test_distributed_*`, `tests/integration/test_online_ws2_*` still exercise `rs.online.olmoe_ep` and `rs.runtime.distributed_ep`
- `experiments/offline/*` wrappers still delegate to `experiments/poc_line1/*`

### 6. Old path final disposition

| Old Path | Intended Destination | Current Status |
| --- | --- | --- |
| `src/rs/online/olmoe_ep` | `legacy/hf_olmoe_ep_harness` or deprecated shim | still live old implementation |
| `src/rs/runtime/distributed_ep` | `legacy/hf_olmoe_ep_harness` or deprecated shim | still live old implementation |
| `experiments/distributed` | `legacy/hf_olmoe_ep_harness` or delete after migration | still live old experiment tree |
| `experiments/poc_line1` | `experiments/offline` + `src/rs/runtime/offline` + `src/rs/scheduling` | still live historical experiment tree |
| `src/rs/scheduler` | `src/rs/scheduling` | partially migrated, old tree still present |
| `src/rs/evaluation` | `src/rs/runtime/offline` + `src/rs/evaluation` split | not fully migrated |
| `src/rs/trace` | `src/rs/runtime/offline/trace` | not fully migrated |
| `integrations/megatron_ep` | thin shim / experiment entry surface only | partially migrated |

### 7. Immediate cleanup summary

Completed in this pass:

- fixed `src/rs/scheduling/matching.py` self-import
- fixed `src/rs/runtime/online/megatron_ep/phase/p0.py`
- fixed `src/rs/runtime/online/megatron_ep/phase/p1.py`
- removed empty stale directories and caches

Still pending for full closure:

- move distributed control-plane agreement out of `src/rs/scheduling/policy/agreement.py`
- remove formal dependence on `experiments/poc_line1`
- cut formal reliance on `rs.online.olmoe_ep` and `rs.runtime.distributed_ep`
- reduce `integrations/megatron_ep` to wrappers plus experiment entrypoints only
