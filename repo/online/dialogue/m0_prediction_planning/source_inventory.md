# M0 Prediction + Planning Source Inventory

Generated before formal edits. This file records the pre-refactor surface area and the intended treatment for each source group.

## Prediction

### Predictor definitions

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/online/megatron_ep/prediction/simple_predictors.py` | Online traffic-matrix predictors: `zero_hint`, `copy_current_dispatch`, `history_ema` | `MIGRATE` | Real deployable baselines; move behind one formal predictor API. |
| `src/rs/runtime/offline/prediction/history_predictor.py` | Offline history predictor implementation | `MIGRATE` | Shares semantics with online history predictor but currently separate type. |
| `src/rs/runtime/offline/prediction/linear_predictor.py` | Offline ridge-linear traffic predictor | `MIGRATE` | CPU-only implementation; should become unified predictor with offline-only flag. |
| `src/rs/runtime/online/megatron_ep/prediction/gate_replay_predictor.py` | Expert-route style predictor contract + mock implementation | `MIGRATE` | Keep as expert-route predictor path, but convert to formal `PredictionResult`. |
| `src/rs/runtime/online/megatron_ep/target_planning/predictor.py` | Two-horizon predictor wrapper and factory logic | `MIGRATE` | Current runtime call path; should use registry instead of ad hoc branching. |

### Predictor registry / factory / construction

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/online/megatron_ep/target_planning/predictor.py` | Runtime predictor selection for target planning | `MIGRATE` | Current runtime factory. |
| `src/rs/runtime/offline/prediction/__init__.py` | Offline public prediction surface | `THIN_SHIM` | Re-export new prediction package after migration. |
| `experiments/offline/train_fate_style_predictor.py` | Private predictor builder | `MIGRATE` | Should call `PredictionRegistry.create`. |

### Online predictor callers

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/online/megatron_ep/target_planning/planner_service.py` | Background target planning path | `MIGRATE` | First-class runtime caller for M0. |
| `src/rs/runtime/online/megatron_ep/lifecycle.py` | Current-window and target-window planning glue | `MIGRATE` | Import-only and adapter-level changes are allowed this round. |
| `src/rs/runtime/online/megatron_ep/control/p2_provider.py` | Reads predictor metadata from runtime state | `RETAINED_INTERNAL` | Metadata reader only; keep behavior stable. |

### Offline predictor callers

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/offline/replay_unified.py` | Replay-to-planning bridge | `MIGRATE` | Must use formal prediction/planning interfaces. |
| `src/rs/runtime/offline/prediction/evaluation.py` | Rolling predictor replay and metrics | `MIGRATE` | Becomes formal evaluation caller. |
| `experiments/offline/evaluate_fate_style_predictor.py` | Prediction evaluation runner | `MIGRATE` | Should stop importing legacy offline predictor internals directly. |
| `experiments/offline/run_prediction_replay_suite.py` | Prediction closure fixture runner | `MIGRATE` | Candidate parity runner for M0. |
| `experiments/offline/run_replay_fixture_policy_suite.py` | Planning and predictor replay suite | `MIGRATE` | Candidate parity runner for M0. |

### Expert-route prediction path

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/online/megatron_ep/prediction/gate_replay_predictor.py` | Expert-first predictor result | `MIGRATE` | Convert to formal `ExpertRoutePrediction`. |
| `src/rs/runtime/online/megatron_ep/prediction/expert_trace.py` | Route aggregation / source expert counts | `RETAINED_INTERNAL` | Shared helper; keep as internal implementation for now. |
| `src/rs/runtime/online/megatron_ep/prediction/expert_trace_capture.py` | Artifact capture | `OUT_OF_SCOPE` | Not part of formal API in M0. |

### Traffic-matrix prediction path

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/online/megatron_ep/prediction/simple_predictors.py` | Online matrix predictors | `MIGRATE` | Becomes implementation behind `rs.prediction`. |
| `src/rs/runtime/offline/prediction/history_predictor.py` | Offline history predictor | `MIGRATE` | Merge under one predictor API. |
| `src/rs/runtime/offline/prediction/linear_predictor.py` | Offline linear predictor | `MIGRATE` | Registry should mark as offline-only. |
| `src/rs/runtime/offline/prediction/dispatch_predictor.py` | Legacy forecast builder including perfect-trace leak path | `THIN_SHIM` | Keep only compatibility conversion where required by old offline flows. |

### Route-to-traffic conversion path

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/online/megatron_ep/prediction/expert_to_traffic.py` | Existing conversion semantics and audits | `MIGRATE` | Source of truth for `RouteToTrafficMapper`. |
| `src/rs/runtime/online/megatron_ep/prediction/expert_evaluation.py` | Uses reconstructed traffic | `MIGRATE` | Evaluator should consume unified prediction result. |

### Prediction evaluator

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/offline/prediction/evaluation.py` | Traffic predictor replay metrics | `MIGRATE` | Base for formal `PredictionEvaluator`. |
| `src/rs/runtime/online/megatron_ep/prediction/expert_evaluation.py` | Expert-route metrics | `MIGRATE` | Merge into formal evaluator. |
| `src/rs/runtime/online/megatron_ep/prediction/audit.py` | Runtime prediction audit row | `RETAINED_INTERNAL` | Runtime artifact formatting only. |

### Prediction duplication summary

- Online traffic predictor interface: `TrafficPredictor.predict(prediction_input, current_dispatch_matrix)`
- Offline traffic predictor interface: `predict_matrix(sample)` with separate fit/export model lifecycle
- Runtime two-horizon wrapper with its own predictor-name branching
- Expert-route mock predictor with separate result contract
- Legacy offline `build_dispatch_forecast()` helper mixing deployable hints and oracle truth

## Planning

### Planner / policy interfaces

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/scheduling/legacy_interfaces.py` | Legacy policy protocols | `RETAINED_INTERNAL` | Implementation layer only after M0. |
| `src/rs/scheduling/unified_interface.py` | Current public scheduling request + policy adapter | `MIGRATE` | Main source for formal request/planner adapter logic. |
| `src/rs/scheduling/registry.py` | Policy resolution | `THIN_SHIM` | Re-export only after introducing `PlannerRegistry`. |
| `src/rs/scheduling/catalog.py` | Canonical algorithm inventory | `MIGRATE` | Fold into formal planner registry metadata. |
| `src/rs/scheduling/algorithm_catalog.py` | Historical algorithm metadata catalog | `DELETE` | Duplicates catalog role; replace with migration record. |
| `src/rs/scheduling/public_catalog.py` | Public policy descriptor layer | `DELETE` | Duplicates catalog role; replace with registry metadata queries. |

### Local planners / baselines

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/scheduling/phase_local/*.py` | Local/baseline deployable policies | `RETAINED_INTERNAL` | Keep algorithms; expose through `rs.planning.local` / `baseline`. |
| `src/rs/scheduling/baselines/*.py` | Reference baselines | `RETAINED_INTERNAL` | Formal family should report `baseline`. |

### Joint planners

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/scheduling/multiphase/tier1.py` | Main joint heuristic family | `RETAINED_INTERNAL` | Use as implementation behind formal joint planners. |
| `src/rs/scheduling/multiphase/safe_joint.py` | Safe compare wrapper | `MIGRATE` | Behavior should move under formal `PlannerSelector.COMPARE`. |
| `src/rs/scheduling/multiphase/routersense_lookahead.py` | Prepared-window / lookahead planning | `RETAINED_INTERNAL` | Internal implementation only. |

### Exact planners

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/scheduling/reference/exact_small_instance.py` | Exact small-instance reference | `RETAINED_INTERNAL` | Formal family `exact_joint` or `exact_local` via adapter metadata. |
| `src/rs/scheduling/reference/birkhoff_von_neumann_fluid.py` | Local oracle-like reference | `RETAINED_INTERNAL` | Formal family `exact_local` / `baseline` depending canonical mapping. |
| `src/rs/scheduling/reference/oracle_guided.py` | Unsupported oracle helpers | `OUT_OF_SCOPE` | Reference-only reporting path. |

### Selector / compare logic

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/scheduling/multiphase/safe_joint.py` | Raw-U vs paired-B comparison | `MIGRATE` | Preserve math, move selection API to `PlannerSelector`. |
| `src/rs/runtime/online/megatron_ep/target_planning/planner_service.py` | Raw-U / paired-B host-selection | `MIGRATE` | Must use formal selector + estimator. |
| `src/rs/runtime/online/megatron_ep/lifecycle.py` | Ad hoc `build_policy(...).plan(...)` compare paths | `MIGRATE` | Import surface only in M0. |

### Scheduling / plan-like contracts

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/scheduling/unified_interface.py` | `SchedulingRequest`, `PolicyOptions` | `MIGRATE` | Replace with formal `PlanningRequest`, `PlanningWeights`, `PlanningConstraints`. |
| `src/rs/scheduling/contracts.py` | `LogicalSchedulePlan`, `LogicalWave`, `PreparedWindowPlan`, `MultiPhaseSchedulingProblem` | `RETAINED_INTERNAL` | Internal legacy plan/problem representation behind adapters. |
| `src/rs/runtime/online/megatron_ep/target_planning/contracts.py` | `TargetLayerPreparedJointPlan`, `TwoHorizonPrediction` | `MIGRATE` | Preserve runtime state/output compatibility with adapters. |
| `src/rs/runtime/offline/replay_unified.py` | `ReplayWindow`, `PlanningHint`, `ExecutionTruth` | `MIGRATE` | Replace with core contracts and formal planner path. |

### Formal callers to migrate this round

| Path | Role | Status | Notes |
| --- | --- | --- | --- |
| `src/rs/runtime/online/megatron_ep/target_planning/planner_service.py` | Runtime target planning | `MIGRATE` | Required M0 caller. |
| `src/rs/runtime/offline/replay_unified.py` | Offline replay engine | `MIGRATE` | Required M0 caller. |
| `experiments/offline/evaluate_fate_style_predictor.py` | Prediction evaluation | `MIGRATE` | Required M0 caller. |
| `experiments/offline/run_prediction_replay_suite.py` | Prediction replay parity | `MIGRATE` | Required M0 caller. |
| `experiments/offline/run_replay_fixture_policy_suite.py` | Planning replay parity | `MIGRATE` | Required M0 caller. |

### Planning duplication summary

- Metadata duplication: `catalog.py`, `algorithm_catalog.py`, `public_catalog.py`
- Factory duplication: `registry.py`, `unified_interface.build_policy()`, runtime target-planning selection logic
- Request duplication: `SchedulingRequest`, `MultiPhaseSchedulingProblem`, offline `PlanningHint`/`ReplayWindow`
- Selection duplication: safe-joint wrapper and runtime host-selection code
- Naming duplication: canonical IDs coexist with `B/U/O_*`, online adapter names, and deprecated aliases

## Initial treatment summary

### KEEP / RETAINED_INTERNAL

- `src/rs/scheduling/contracts.py`
- `src/rs/scheduling/phase_local/*`
- `src/rs/scheduling/multiphase/tier1.py`
- `src/rs/scheduling/reference/*`
- `src/rs/runtime/online/megatron_ep/prediction/expert_trace.py`

### MIGRATE

- `src/rs/runtime/online/megatron_ep/prediction/simple_predictors.py`
- `src/rs/runtime/offline/prediction/history_predictor.py`
- `src/rs/runtime/offline/prediction/linear_predictor.py`
- `src/rs/runtime/online/megatron_ep/prediction/gate_replay_predictor.py`
- `src/rs/runtime/online/megatron_ep/target_planning/predictor.py`
- `src/rs/runtime/online/megatron_ep/target_planning/planner_service.py`
- `src/rs/runtime/offline/replay_unified.py`
- `src/rs/runtime/offline/prediction/evaluation.py`
- `src/rs/scheduling/unified_interface.py`
- `src/rs/scheduling/catalog.py`
- `src/rs/scheduling/multiphase/safe_joint.py`

### THIN_SHIM

- `src/rs/runtime/offline/prediction/__init__.py`
- `src/rs/runtime/offline/prediction/dispatch_predictor.py`
- `src/rs/scheduling/registry.py`

### DELETE

- `src/rs/scheduling/algorithm_catalog.py`
- `src/rs/scheduling/public_catalog.py`

### OUT_OF_SCOPE

- Runtime artifact writers, observer export paths, diagnostics-only experiment code
- GPU/NCCL/online execution backends
- Lifecycle event order and queue / thread semantics
