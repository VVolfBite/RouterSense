# M0 Closure 2 Full Module Audit

## Scope

Read modules:

- `src/rs/core/contracts/prediction.py`
- `src/rs/core/contracts/planning.py`
- `src/rs/prediction/**`
- `src/rs/planning/**`
- `src/rs/runtime/offline/**`
- `src/rs/runtime/online/megatron_ep/target_planning/**`
- `src/rs/runtime/online/megatron_ep/lifecycle.py` prediction/planning call sites
- targeted tests under `tests/contract`, `tests/unit`, `tests/offline`, `tests/test_architecture_dependencies.py`

## Public classes and functions

### Core prediction contracts

- `PredictionIdentity`
- `ExpertRouteContext`
- `TrafficHistoryContext`
- `PredictionHint`
- `ExpertRoutePrediction`
- `RankedExpertRoutes`
- `PredictionResult`

Primary public methods:

- `to_dict()`
- `validate()`

### Core planning contracts

- `PlanningIdentity`
- `PlanningTraffic`
- `PlanningTopology`
- `PlanningConstraints`
- `PlanningWeights`
- `PlanningRequest`
- `PlannedFlow`
- `PlanWave`
- `WindowPlan`
- `PlanScore`

Primary public methods:

- `validate()`
- `to_dict()`
- `semantic_payload()`
- `semantic_digest()`
- `identity_digest()` on `PlanningRequest`
- `audit_digest()` on `WindowPlan`

### Prediction package

- `Predictor`
- `PredictorSpec`
- `TrafficPredictionTrainingSample`
- `TrainableTrafficPredictor`
- `PredictionRegistry.specs()`
- `PredictionRegistry.create(...)`
- `resolve_predictor_id(...)`
- `PredictionEvaluator.evaluate(...)`
- `RouteToTrafficMapper.map(...)`
- `RouteToTrafficMapper.map_ranked(...)`
- predictors:
  - `ZeroTrafficPredictor`
  - `CopyCurrentTrafficPredictor`
  - `HistoryTrafficPredictor`
  - `LinearTrafficPredictor`
  - `MockGateReplayExpertRoutePredictor`

### Planning package

- `Planner`
- `PlannerSpec`
- `PlannerPolicyConfig`
- `PlannerRegistry.specs()`
- `PlannerRegistry.create(...)`
- `PlanningCostModel`
- `PlanEstimator`
- `CommonCorePlanEstimator.estimate(...)`
- `PlannerSelectionMode`
- `PlanningSelectionError`
- `SelectedPlan`
- `PlannerSelector.select(...)`
- `PlannerSelector.select_prebuilt(...)`

Private compatibility surface retained:

- `rs.planning._legacy_runtime`
- `rs.planning.runtime_compat`
- `rs.planning.api.to_legacy_request(...)`
- `rs.planning.api.to_logical_plan(...)`
- `rs.planning.api.from_logical_plan(...)`
- `rs.planning.api.LegacyPlannerAdapter`

## Formal callers

### Predictor construction points

- `runtime/online/megatron_ep/lifecycle.py`
  - `_build_online_predictor()`
- `runtime/online/megatron_ep/target_planning/predictor.py`
  - `SharedTwoHorizonPredictor.__init__`
- `runtime/offline/prediction/evaluation.py`
  - `rolling_predictor_records(...)`
- tests:
  - `tests/contract/test_prediction_api.py`
  - `tests/offline/test_prediction_planning_parity.py`

### Planner construction points

- `runtime/offline/replay_unified.py`
  - `ReplayEngine.execute(...)`
- `runtime/online/megatron_ep/target_planning/planner_service.py`
  - `TargetLayerPlannerService._build_target_plan(...)`
- tests:
  - `tests/contract/test_planning_api.py`
  - `tests/unit/test_planner_selector.py`
  - `tests/contract/megatron_ep/test_target_planner_service.py`

### PlannerSelector call sites

- `src/rs/runtime/online/megatron_ep/target_planning/planner_service.py`
  - `_select_candidate_plans(...)`
- `tests/unit/test_planner_selector.py`
- `tests/contract/megatron_ep/test_target_planner_service.py`

### Estimator call sites

- `src/rs/planning/selection.py`
- `src/rs/runtime/online/megatron_ep/target_planning/planner_service.py`
- `tests/unit/test_plan_estimator.py`
- `tests/contract/test_truth_hint_isolation.py`

## Legacy callers and retained compatibility surfaces

### Runtime current-window legacy planning path

- `runtime/online/megatron_ep/lifecycle.py`
  - still uses private compatibility builders from `rs.planning._legacy_runtime`
  - still builds legacy logical scheduling requests for current-window path
  - not formal public API anymore

### Offline legacy scheduling bridge

- `runtime/offline/runner.py`
  - still uses private compatibility builders from `rs.planning._legacy_runtime`
  - this is retained legacy study surface, not formal planning API

### Alias/legacy lookup only

- `rs.planning.runtime_compat`
  - `ResolvedAlgorithmId`
  - `resolve_algorithm_id`
  - `resolve_phase_policy`
  - `supported_phase_policies`

## Digest generation points

Stable digest points reviewed:

- `PlanningRequest.semantic_digest()`
- `PlanningRequest.identity_digest()`
- `WindowPlan.semantic_digest()`
- `WindowPlan.audit_digest()`
- `runtime/offline/replay_unified.py`
  - `execution_truth_digest(...)`
  - `CanonicalBucketizer.digest(...)` on planning-visible tasks
- `runtime/online/megatron_ep/target_planning/predictor.py`
  - `TwoHorizonPrediction.matrix_digest`
- `runtime/online/megatron_ep/lifecycle.py`
  - runtime state digests use `stable_hash(...)`
- `runtime/online/megatron_ep/target_planning/planner_service.py`
  - `logical_plan_digest`
  - `target_problem_digest`

Removed unstable digest point:

- built-in `hash(...)` in `target_planning/predictor.py`

## Truth / hint access points

### Truth holders

- `runtime/offline/replay_unified.py`
  - `ReplayWindow.p2_truth_rows`
  - `ExecutionTruth`
  - `execution_truth_digest(...)`

### Hint holders

- `core/contracts/prediction.py`
  - `PredictionHint.target_dispatch_rows`
- `runtime/offline/replay_unified.py`
  - `PlanningHint.p2_hint_rows`
- `runtime/online/megatron_ep/target_planning/predictor.py`
  - `TwoHorizonPrediction.matrix_rows`

### Planning-visible truth/hint boundaries

- `runtime/offline/replay_unified.py`
  - `build_multiphase_problem(...)` now writes hint, not truth, into forecast
  - `bucketize_planning_request(...)` uses only `PlanningRequest`
- `core/contracts/planning.py`
  - `PlanningRequest.semantic_payload()` uses only planner-visible semantics

## Exception and fallback paths

### Prediction

- `PredictionRegistry.create(...)`
  - rejects invalid `usage`
  - rejects `test_only` outside `usage="test"`
  - rejects `offline_only` in runtime usage
- `LinearTrafficPredictor.predict(...)`
  - raises if `fit()` not called
- `PredictionEvaluator.evaluate(...)`
  - returns invalid evaluation on shape mismatch or incomplete expert-route shape

### Planning

- `PlannerSelector.select_prebuilt(...)`
  - raises `PlanningSelectionError` when selected single-mode plan is invalid
  - raises `PlanningSelectionError` when both compare candidates are invalid
- `CommonCorePlanEstimator.estimate(...)`
  - returns invalid `PlanScore` on digest mismatch, invalid flow, invalid wave, port conflict

### Runtime target planning

- `TargetLayerPlannerService._build_target_plan(...)`
  - raises if `host_select` missing paired local policy id

## Tests covering this surface

### Contract tests

- `tests/contract/test_prediction_api.py`
- `tests/contract/test_planning_api.py`
- `tests/contract/test_truth_hint_isolation.py`
- `tests/contract/megatron_ep/test_lifecycle_prediction_adapter.py`
- `tests/contract/megatron_ep/test_target_planner_service.py`
- `tests/contract/megatron_ep/test_target_plan_contracts.py`
- `tests/contract/megatron_ep/test_two_horizon_prediction.py`

### Unit tests

- `tests/unit/test_planner_selector.py`
- `tests/unit/test_plan_estimator.py`

### Offline parity / smoke

- `tests/offline/test_prediction_planning_parity.py`
- `experiments/run_offline_replay.py --config tests/fixtures/configs/minimal_offline.yaml`

## Prompt-external issues found and fixed

- `PlanningTraffic.to_dict()` validation path assumed `world_size` was always passed; this broke digest generation.
- Formal planning tests still used root-relative fixture paths after the repo nesting layout; offline parity path needed `RS/tests/...`.
- Legacy planners emit phase name `p2_next_dispatch`; contract validation initially only allowed `p2_next_dispatch_forecast`.
