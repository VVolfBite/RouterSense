## Pre-Evaluation Cutover Manifest

This manifest records the main repository cutover decisions that define the
formal pre-evaluation mainline.

| Old path | New path | Status |
| --- | --- | --- |
| `integrations/megatron_ep/` | `legacy/historical_poc/integrations/` | moved to legacy |
| `experiments/poc_line1/` | `legacy/historical_poc/experiments_poc_line1/` | moved to legacy |
| `experiments/distributed/` | `legacy/historical_poc/experiments_distributed/` | moved to legacy |
| `experiments/legacy/` | `legacy/historical_poc/experiments_legacy/` | moved to legacy |
| `experiments/offline/run_multiphase_reference.py` | `legacy/historical_poc/experiments_offline/run_multiphase_reference.py` | moved to legacy |
| `experiments/offline/run_scheduler_ablation.py` | `legacy/historical_poc/experiments_offline/run_scheduler_ablation.py` | moved to legacy |
| `experiments/offline/compare_prediction_inputs.py` | `legacy/historical_poc/experiments_offline/compare_prediction_inputs.py` | moved to legacy |
| `experiments/offline/exp_router_prediction.py` | `legacy/historical_poc/experiments_offline/exp_router_prediction.py` | moved to legacy |
| `experiments/offline/exp_calibrated_schedule.py` | `legacy/historical_poc/experiments_offline/exp_calibrated_schedule.py` | moved to legacy |
| `experiments/offline/fit_ep_cost_model.py` | `legacy/historical_poc/experiments_offline/fit_ep_cost_model.py` | moved to legacy |
| `experiments/online/bench_native_ep.py` | `legacy/historical_poc/experiments_online/bench_native_ep.py` | moved to legacy |
| `experiments/online/bench_scheduled_ep.py` | `legacy/historical_poc/experiments_online/bench_scheduled_ep.py` | moved to legacy |
| `experiments/ablation/configs/*.yaml` | `configs/experiment/ablation/*.yaml` | moved to config tree |
| `analysis/` | `scripts/metrics/`, `scripts/plot/` | moved |
| `tools/archive/` | `scripts/maintenance/archive/` | moved |
| `archives/` | `legacy/historical_poc/archives/` | moved to legacy |
| `src/rs/online/` | `legacy/historical_poc/src_rs_legacy/online/` | moved to legacy |
| `src/rs/runtime/distributed_ep/` | `legacy/historical_poc/src_rs_legacy/runtime/distributed_ep/` | moved to legacy |
| `src/rs/contracts/` | `legacy/historical_poc/src_rs_legacy/contracts/` | moved to legacy |
| `src/rs/offline/` | `legacy/historical_poc/src_rs_legacy/offline/` | moved to legacy |
| `src/rs/evaluation/` | `legacy/historical_poc/src_rs_legacy/evaluation/` | moved to legacy |
| `src/rs/scheduler/` | `legacy/historical_poc/src_rs_legacy/scheduler/` | moved to legacy |
| `src/rs/trace/` | `legacy/historical_poc/src_rs_legacy/trace/` | moved to legacy |
| `src/rs/legacy/` | `legacy/historical_poc/src_rs_legacy/legacy_pkg/` | moved to legacy |
| tracked local/current deploy inventories | ignored local files only | removed from Git |

Formal mainline after cutover:

- `src/rs/core/`
- `src/rs/scheduling/`
- `src/rs/runtime/offline/`
- `src/rs/runtime/online/megatron_ep/`
- `experiments/offline/`
- `experiments/online/`
- `configs/`
- `scripts/`
- `docs/`
- `archive/`
