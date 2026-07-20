# Prediction Final Design

## Scope

This stage keeps the predictor family intentionally narrow:

- `none` / zero-hint
- `copy_current_dispatch`
- `history_ema`
- optional diagnostic-only oracle/shuffled controls

No faithful FATE gate replay is claimed here.

## Runtime Lifecycle

For current layer `L`:

- actual `P0(L)` is gathered exactly once
- prediction for `P0(L+1)` is generated during `P0(L)`
- that prediction is consumed immediately during current-layer joint-window planning
- the same prediction acts as future pressure for current `P1(L)`
- when `P0(L+1)` begins, the old prediction for target `L+1` is audited before the new target `L+2` prediction is created

## State Model

Predictions are keyed by runtime window identity and kept bounded.

The runtime only retains:

- current actual layer traffic
- old prediction targeting current layer
- new prediction targeting next layer
- current stored joint plan

It does not retain unbounded full-layer traffic history in hot runtime state.

## Confidence

Prediction confidence is propagated through:

- prediction state
- runtime joint plan
- score component
- diagnostics

Confidence is applied once. The old double-scaling path is no longer the intended semantics.

## Consumption

Prediction is not a standalone runtime phase.

It contributes only through one explicit future-pressure component inside joint planning. The current runtime target is:

- prediction should change score when it carries real structure
- prediction should not be silently discarded
- prediction should not trigger any extra all-gather beyond the existing compact P0 summary gather

## Current Selection Rule

The online runtime is prepared to compare online-eligible predictors, but this round does not expand predictor families. The next GPU step is to validate:

- zero extra collectives
- correct source/target layer lifecycle
- confidence preservation
- actual scheduling benefit relative to zero-hint
