# M0-CLOSURE-2 Report

## Baseline

- branch: `convergence/m0-prediction-planning`
- starting commit: `50b76d7`

## Issues fixed from prompt

1. `PlannerSelector` invalid-plan selection
   - fixed in `src/rs/planning/selection.py`
   - compare now uses explicit valid/invalid rules
   - both invalid now raise `PlanningSelectionError`

2. `PredictionEvaluator` incomplete input handling
   - fixed in `src/rs/prediction/evaluation.py`
   - traffic shape mismatch now returns invalid evaluation
   - expert-route token/top-k mismatch now returns invalid evaluation
   - metrics now carry `valid`, `reason`, `predicted_shape`, `truth_shape`

3. `RouteToTrafficMapper` invalid rank / owner acceptance
   - fixed in `src/rs/prediction/route_to_traffic.py`
   - rejects negative `source_rank`
   - rejects negative / out-of-range owner rank
   - rejects world-size mismatch in ranked input
   - rejects invalid expert id
   - keeps diagonal/self traffic at zero

4. Linear predictor feature semantics
   - fixed in `src/rs/prediction/traffic_matrix/predictors.py`
   - formal linear predictor now uses real `current_return_rows`
   - training and inference use the same feature builder
   - parity test added against legacy FATE-style linear predictor

5. `CommonCorePlanEstimator` legacy duration influence
   - fixed in `src/rs/planning/estimation.py`
   - no longer trusts `legacy_makespan`
   - no longer trusts `wave.estimated_duration`
   - selection estimate now depends only on flows + request + cost model

6. Formal planning API still exposing legacy scheduling objects
   - fixed by removing runtime builder helpers from `rs.planning` exports
   - legacy builders downgraded to private `rs.planning._legacy_runtime`

7. `test_only` predictor still directly constructible
   - fixed in `src/rs/prediction/registry.py`
   - `PredictionRegistry.create(..., usage=...)` added
   - rules:
     - `test_only`: only `usage="test"`
     - `offline_only`: `offline` / `test`, not `runtime`
     - deployable: all three

8. Incomplete contract validation
   - fixed in:
     - `src/rs/core/contracts/prediction.py`
     - `src/rs/core/contracts/planning.py`
   - prediction and planning contracts now validate matrix shape, non-negativity, ranges, confidence, topology, constraints, weights, and identity basics

9. Semantic digest still mixed predictor provenance
   - fixed in `src/rs/core/contracts/planning.py`
   - `PlanningRequest.semantic_digest()` now excludes:
     - `predictor_id`
     - `hint_type`
     - `confidence`
     - source/target provenance
     - runtime identity
   - only planner-visible matrix semantics remain

10. H2 identity wrong
   - fixed in `src/rs/runtime/online/megatron_ep/target_planning/predictor.py`
   - H2 now has:
     - `source_layer_id = H1.target_layer_id`
     - `target_layer_id = source+2`
   - target-planning formal request now aligns planning identity with H2 semantics

11. safe-disabled metadata pretended paired local plan existed
   - fixed in `src/rs/runtime/online/megatron_ep/target_planning/contracts.py`
   - fixed in `src/rs/runtime/online/megatron_ep/target_planning/planner_service.py`
   - prepared target plan now records:
     - `safe_projection_mode`
     - `raw_u_plan_was_built/scored/selected`
     - `paired_b_plan_was_built/scored/selected`
   - disabled mode no longer implies paired local build

12. runtime-safe policy overlap with formal compare owner
   - formal target-layer prepared path now uses `PlannerSelector.COMPARE`
   - current-window lifecycle path still has old host-projected safe logic and is listed in remaining legacy surface, not claimed as converged

13. built-in `hash(...)` in formal digest path
   - fixed in `src/rs/runtime/online/megatron_ep/target_planning/predictor.py`
   - replaced with stable SHA-256 based digest helper
   - cross-process stability tests added

## Prompt-external issues found and fixed

1. `PlanningTraffic.to_dict()` validation path assumed `world_size` was always passed and broke `semantic_digest()`.
2. offline parity fixture path still pointed to `tests/...` instead of `RS/tests/...` in this workspace layout.
3. legacy planners emit phase name `p2_next_dispatch`; validation initially rejected it although it is a real retained legacy internal phase label.

## Unfixed items

- current-window runtime path in `lifecycle.py` still uses private legacy scheduling builders
- offline study runner in `runtime/offline/runner.py` still uses private legacy scheduling builders
- deployable expert-route predictor is still not migrated; registry now marks `mock_gate_replay` as `test_only`

These are recorded in `remaining_legacy_surface.md`.

## Formal API after this round

### Prediction

- contracts: `src/rs/core/contracts/prediction.py`
- public package: `src/rs/prediction`
- runtime/offline construction gate: `PredictionRegistry.create(..., usage=...)`

### Planning

- contracts: `src/rs/core/contracts/planning.py`
- public package: `src/rs/planning`
- selection owner: `PlannerSelector`
- cost owner: `CommonCorePlanEstimator`

## Tests

- `python -m compileall RS/src RS/experiments RS/tests`
- targeted pytest:
  - `60 passed, 2 warnings`
- offline smoke:
  - `experiments/run_offline_replay.py --config tests/fixtures/configs/minimal_offline.yaml --output-dir outputs/offline/m0_final_closure_smoke`
  - `run_valid = true`
  - `audit_invalid_count = 0`

## Merge status

- M0 prediction/planning semantic closure conditions requested in this round are satisfied
- remaining legacy surface is explicitly isolated and documented
