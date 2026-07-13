## Core contracts

- Prediction authority: `src/rs/core/contracts/prediction.py`
- Planning authority: `src/rs/core/contracts/planning.py`

## Prediction authority surface

- Public API: `src/rs/prediction/api.py`
- Registry: `src/rs/prediction/registry.py`
- Traffic predictors: `src/rs/prediction/traffic_matrix/predictors.py`
- Expert-route predictor: `src/rs/prediction/expert_route/predictors.py`
- Route mapper: `src/rs/prediction/route_to_traffic.py`
- Evaluator: `src/rs/prediction/evaluation.py`

### Migrated callers

- Online predictor factory: `src/rs/runtime/online/megatron_ep/target_planning/predictor.py`
- Lifecycle predictor creation and dispatch hint path: `src/rs/runtime/online/megatron_ep/lifecycle.py`
- Offline rolling prediction evaluation: `src/rs/runtime/offline/prediction/evaluation.py`

## Planning authority surface

- Public API: `src/rs/planning/api.py`
- Registry: `src/rs/planning/registry.py`
- Selector: `src/rs/planning/selection.py`
- Estimator + cost model: `src/rs/planning/estimation.py`
- Legacy alias normalization: `src/rs/planning/legacy_aliases.py`
- Runtime compatibility shim: `src/rs/planning/runtime_compat.py`

### Migrated callers

- Online target planner service: `src/rs/runtime/online/megatron_ep/target_planning/planner_service.py`
- Runtime algorithm resolution imports: `src/rs/runtime/online/megatron_ep/runtime.py`
- Pending-window phase policy adapter: `src/rs/runtime/online/megatron_ep/pending_window/policy_adapter.py`
- Lifecycle planning imports: `src/rs/runtime/online/megatron_ep/lifecycle.py`
- Offline replay engine entry: `src/rs/runtime/offline/replay_unified.py`
