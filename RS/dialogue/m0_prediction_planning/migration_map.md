## Prediction

- `rs.runtime.online.megatron_ep.prediction.simple_predictors`
  - `ZeroHintPredictor` -> `rs.prediction.traffic_matrix.ZeroTrafficPredictor`
  - `CopyCurrentDispatchPredictor` -> `rs.prediction.traffic_matrix.CopyCurrentTrafficPredictor`
  - `HistoryEMATrafficPredictor` -> `rs.prediction.traffic_matrix.HistoryTrafficPredictor`
- `rs.runtime.offline.prediction.linear_predictor`
  - ridge linear logic migrated into `rs.prediction.traffic_matrix.LinearTrafficPredictor`
- `rs.runtime.online.megatron_ep.prediction.gate_replay_predictor`
  - replay-driven expert-route predictor migrated into `rs.prediction.expert_route.MockGateReplayExpertRoutePredictor`
- traffic reconstruction helpers
  - converged into `rs.prediction.route_to_traffic.RouteToTrafficMapper`

## Planning

- Public scheduling entrypoints converged into `rs.planning.registry.PlannerRegistry`
- Selection logic converged into `rs.planning.selection.PlannerSelector`
- Cross-planner score normalization converged into `rs.planning.estimation.CommonCorePlanEstimator`
- Runtime policy option/request construction converged into `rs.planning.api.PlannerPolicyConfig`, `build_runtime_policy(...)`, and `build_runtime_request_from_problem(...)`
- Legacy scheduling implementations remain internal behind `rs.planning.api.LegacyPlannerAdapter`

## Runtime caller migration

- Online predictor factory now calls `PredictionRegistry.create(...)`
- Online target planning now calls `PlannerRegistry.create(...)` and `PlannerSelector.select(...)`
- Offline replay now builds `PlanningRequest` and calls formal planner API
- Offline prediction evaluation now builds `TrafficHistoryContext` and calls formal predictor API

## Compatibility boundaries kept in M0

- `rs.scheduling` retained as internal implementation layer
- `rs.planning.runtime_compat` retained as a thin shim for unresolved runtime import sites
- runtime output adapters still translate `PredictionResult` and `WindowPlan` back to legacy metadata/object shapes where the surrounding runtime still expects them
- lifecycle prediction adapter now uses a typed runtime compatibility object rather than a dict
