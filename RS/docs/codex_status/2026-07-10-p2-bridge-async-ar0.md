# 2026-07-10 P2 Bridge And Async AR0

## Scope

This round stayed CPU-only and focused on four areas:

1. fix `policy_explain` causality
2. fix online P2 bridge semantics
3. regenerate corrected P2-consumption diagnosis
4. push async-release from abstract skeleton to AR0 runtime-plan + AR1 interface

## Main Corrections

### Policy explain

- safe makespan is no longer paired with raw-U order by mistake
- raw U / paired B / safe selected order and matching are reported separately
- `selected_order`, `selected_matching`, and `bottleneck_edges` now refer to the safe actual choice
- canonical first-service order replaced the old set-difference pseudo order metric
- `critical_edges` was renamed to `top_score_edges`

### Prediction confidence

- prediction confidence is now applied once, not twice
- `outbound_loads()` returns raw remote load
- scoring applies confidence at the prediction component

### Online P2 bridge

- logical `P2(layer L)` is now mapped to runtime `P0(layer L+1)`
- stale logical `P0/P1` prepared edges are ignored for next-layer execution priority
- prepared priority no longer unconditionally overrides live score
- current default mode is `mapped_p2_tiebreak`
- online predictor is now configurable:
  - `none`
  - `copy_current_dispatch`
  - `history_ema`

## Async Release

- AR0 runtime plan builder added:
  - real task ids from real phase tasks
  - explicit event table
  - executable-shaped task payload
- AR1 experimental P2P executor interface added
- real collectives remain disabled and unvalidated

## Outputs

- `outputs/offline/m6s_p2_bridge_async_ar0/`
- `docs/prediction_design_after_4gpu_trace.md`
- `docs/runtime_overhead_and_replay_gap.md`
- `docs/async_release_runtime_plan.md`

## Status Flags

- `gpu_not_run=true`
- `faithful_fate_not_validated=true`
- `async_release_real_collectives_not_validated=true`
