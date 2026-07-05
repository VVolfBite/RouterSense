## Source Of Truth Inventory

This inventory reflects the current pre-evaluation mainline after the repository-cutover pass.
It is based on current imports, active tests, and surviving CLI entrypoints.

### Formal online runtime

Canonical online Megatron EP implementation lives in:

- `src/rs/runtime/online/megatron_ep/_host_impl.py`
- `src/rs/runtime/online/megatron_ep/_lifecycle.py`
- `src/rs/runtime/online/megatron_ep/_observation.py`
- `src/rs/runtime/online/megatron_ep/contracts.py`
- `src/rs/runtime/online/megatron_ep/observer.py`
- `src/rs/runtime/online/megatron_ep/trace_writer.py`
- `src/rs/runtime/online/megatron_ep/p2_provider.py`
- `src/rs/runtime/online/megatron_ep/phase/*`
- `src/rs/runtime/online/megatron_ep/control/*`
- `src/rs/runtime/online/megatron_ep/execution/*`
- `src/rs/runtime/online/megatron_ep/policy_adapter.py`

Stable public entrypoints:

- `src/rs/runtime/online/megatron_ep/host.py`
- `src/rs/runtime/online/megatron_ep/runtime.py`
- `src/rs/runtime/online/megatron_ep/lifecycle.py`
- `src/rs/runtime/online/megatron_ep/artifact_recorder.py`

Status:

- canonical implementation: `src/rs/runtime/online/megatron_ep`
- old `integrations/megatron_ep`: moved to `legacy/historical_poc/integrations`
- formal tests: now import `rs.runtime.online.megatron_ep.*`

### Formal offline runtime

Canonical offline trace / traffic / prediction implementation lives in:

- `src/rs/runtime/offline/trace/*`
- `src/rs/runtime/offline/traffic/*`
- `src/rs/runtime/offline/prediction/*`
- `src/rs/runtime/offline/runner.py`

Status:

- trace collection and prediction helpers are canonical under `runtime/offline`
- `experiments/offline/collect_router_trace.py` and `analyze_cross_layer_prediction.py` now use canonical runtime/offline imports
- historical offline study entrypoints have been moved to `legacy/historical_poc/experiments_offline`
- formal `experiments/offline/` now only contains entrypoints that import canonical `rs.runtime.offline` helpers

### Formal scheduling

Canonical scheduling implementation lives in:

- `src/rs/scheduling/contracts.py`
- `src/rs/scheduling/matching.py`
- `src/rs/scheduling/validation.py`
- `src/rs/scheduling/phase_local/*`
- `src/rs/scheduling/reference/*`
- `src/rs/scheduling/multiphase/*`
- `src/rs/scheduling/policy/*`

Status:

- policy library used by the online runtime is canonical under `src/rs/scheduling/policy`
- pure wire/agreement logic moved out of scheduling into `runtime/online/megatron_ep/control/agreement_wire.py`
- historical `src/rs/scheduler/*` has been moved to `legacy/historical_poc/src_rs_legacy/scheduler`
- formal `scheduling/baselines/birkhoff.py` and `scheduling/reference/{oracle_guided,exact_small_instance}.py` now fail closed with explicit `unsupported` metadata instead of returning placeholder optimality claims

### Old paths and final disposition

| Old path | New / current status | Disposition |
| --- | --- | --- |
| `integrations/megatron_ep` | `legacy/historical_poc/integrations` | legacy |
| `experiments/poc_line1` | `legacy/historical_poc/experiments_poc_line1` | legacy |
| `experiments/distributed` | `legacy/historical_poc/experiments_distributed` | legacy |
| `experiments/legacy` | `legacy/historical_poc/experiments_legacy` | legacy |
| `experiments/ablation/configs/*` | `configs/experiment/ablation/*` | moved |
| `experiments/offline/run_multiphase_reference.py` | `legacy/historical_poc/experiments_offline/run_multiphase_reference.py` | legacy |
| `experiments/offline/run_scheduler_ablation.py` | `legacy/historical_poc/experiments_offline/run_scheduler_ablation.py` | legacy |
| `experiments/offline/compare_prediction_inputs.py` | `legacy/historical_poc/experiments_offline/compare_prediction_inputs.py` | legacy |
| `experiments/offline/exp_router_prediction.py` | `legacy/historical_poc/experiments_offline/exp_router_prediction.py` | legacy |
| `experiments/offline/exp_calibrated_schedule.py` | `legacy/historical_poc/experiments_offline/exp_calibrated_schedule.py` | legacy |
| `experiments/offline/fit_ep_cost_model.py` | `legacy/historical_poc/experiments_offline/fit_ep_cost_model.py` | legacy |
| `experiments/online/bench_native_ep.py` | `legacy/historical_poc/experiments_online/bench_native_ep.py` | legacy |
| `experiments/online/bench_scheduled_ep.py` | `legacy/historical_poc/experiments_online/bench_scheduled_ep.py` | legacy |
| `analysis/` | `scripts/metrics/*`, `scripts/plot/*` | moved |
| `tools/archive/` | `scripts/maintenance/archive/` | moved |
| `archives/` | `legacy/historical_poc/archives` | legacy |
| `src/rs/evaluation` | `legacy/historical_poc/src_rs_legacy/evaluation` | legacy |
| `src/rs/scheduler` | `legacy/historical_poc/src_rs_legacy/scheduler` | legacy |
| `src/rs/trace` | `legacy/historical_poc/src_rs_legacy/trace` | legacy |
| `src/rs/legacy` | `legacy/historical_poc/src_rs_legacy/legacy_pkg` | legacy |
| tracked local/current inventory | `deploy/inventory/*.example`, local ignored copies only | delete from Git |

### Deprecated legacy parking

Historical trees no longer live under the formal `src/rs` namespace. They now live only under legacy parking:

- `legacy/historical_poc/src_rs_legacy/evaluation`
- `legacy/historical_poc/src_rs_legacy/scheduler`
- `legacy/historical_poc/src_rs_legacy/trace`
- `legacy/historical_poc/src_rs_legacy/online`
- `legacy/historical_poc/src_rs_legacy/runtime/distributed_ep`
- `legacy/historical_poc/src_rs_legacy/contracts`
- `legacy/historical_poc/src_rs_legacy/offline`

Current status of these trees:

- they are excluded from the default mainline test path by moving their tests under `tests/legacy`
- formal online tests no longer import them
- formal offline entrypoints no longer import them
