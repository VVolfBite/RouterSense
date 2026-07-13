# Remaining Legacy Surface

## Still retained after M0-CLOSURE-2

### `src/rs/planning/_legacy_runtime.py`

Purpose:

- private runtime/offline bridge to old scheduling request / policy builders

Why retained:

- current-window runtime path in `lifecycle.py` still depends on legacy scheduling request objects
- offline study runner still depends on old logical scheduling bridge
- removing this now would expand into Lifecycle main-path refactor, which is outside M0

Allowed usage:

- runtime current-window path
- offline legacy study path

Not allowed:

- formal `rs.planning` package exports
- new formal callers

### `src/rs/planning/runtime_compat.py`

Purpose:

- alias resolution and legacy phase-policy lookup only

Current contents:

- `ResolvedAlgorithmId`
- `resolve_algorithm_id`
- `resolve_phase_policy`
- `supported_phase_policies`

Not allowed:

- planner construction
- registry duplication
- legacy request construction

### `src/rs/planning/api.py` internal legacy bridge helpers

Still present:

- `to_legacy_request(...)`
- `to_logical_plan(...)`
- `from_logical_plan(...)`
- `LegacyPlannerAdapter`

Status:

- retained internal implementation layer
- not exported from `rs.planning`

## Runtime legacy logic still outside M0

### `src/rs/runtime/online/megatron_ep/lifecycle.py`

Still legacy in current-window path:

- builds legacy scheduling requests through private compatibility helper
- still contains old host-projected runtime-safe current-window comparison path

M0 effect:

- target-layer prepared planning path now uses formal selector semantics
- formal package no longer exports legacy builders

M1 removal target:

- migrate current-window path to formal `PlanningRequest` / `PlannerSelector`
- retire runtime-host safe comparison duplication in current-window path

### `src/rs/runtime/offline/runner.py`

Still legacy in study path:

- uses private compatibility helper for legacy scheduling request objects

M1 removal target:

- move study runner to formal planning request builder or isolate as explicit legacy experiment utility

## Legacy policy ownership

Deployable anti-regression owner should now be:

- `PlannerSelector.COMPARE`

Legacy/reference only:

- `barrier_criticality_runtime_safe`
- posthoc best / reference policies

## Deletion candidates after M1 migration

- `src/rs/planning/_legacy_runtime.py`
- current-window direct use of legacy scheduling request builders in `lifecycle.py`
- offline study direct use of legacy scheduling request builders in `runner.py`
