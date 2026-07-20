# Runtime Joint Async Design

## Formal Runtime Path

The single-node runtime path is:

`Megatron host hook -> RouterSense lifecycle -> canonical Joint/Local plan -> host-projected Safe selection -> local schedule materialization -> batch_isend_irecv`

Execution mode:

- `joint_window_async_p2p`

Formal deployable strategies use canonical planner IDs, for example:

- `fifo_bucket`
- `greedy_bucket`
- `birkhoff_bucket_phase_local`
- `current:p012:local:event:rscf`
- `current:p012:joint:event:rscf`
- `current:p012:local:global:rscf`
- `current:p012:joint:global:rscf`
- `future:p012:joint:global:rscf`

Historical B/U strategy names are not accepted by the formal registry.

## Planning Semantics

For each layer `L` during P0:

1. Build the local `PhaseReadyContext`.
2. Gather one compact global traffic summary.
3. Reconstruct actual `P0(L)`.
4. Audit any prediction targeting `L`.
5. Produce a causal prediction for `P0(L+1)`.
6. Derive `P1(L)` from `P0(L)`.
7. Build one canonical planning request containing actual P0/P1 and advisory P2.
8. Build the configured Joint candidate.
9. When Safe selection is enabled, build the strictly paired Local fallback with the same engine, core and horizon.
10. Apply the same host-feasibility projection to both plans.
11. Select Joint or Local before transport begins.
12. Materialize the selected local P0 schedule.
13. Reuse the selected abstract P1 plan at `before_token_combine`.

P1 does not rebuild prediction, rerun scope selection, regather traffic, or broadcast a second full plan.

## Host-Projected Safe Selection

The runtime compares:

- the host-projected Joint candidate;
- the host-projected same-engine Local fallback.

The selected plan is the host-feasible winner. Fine-grained overlap that the current Megatron hook cannot expose is removed from both candidates before selection.

## Transport Semantics

The runtime claims rank-level asynchronous phase completion:

- `P0_LOCAL_PHASE_COMPLETE`: hidden states and routing probabilities complete locally before `token_dispatch()` returns.
- `P1_LOCAL_PHASE_COMPLETE`: rank-local return receives complete before `token_combine()` continues.

It does not claim per-bucket expert compute overlap, per-expert release, or cancellation after P2P work has started.

## Safety Rules

Before the first P2P operation, the runtime validates:

- global plan digest;
- sequence parity;
- send/receive rows;
- dtype and shape suffix parity;
- receive coverage;
- dedicated P2P group initialization.

A preflight failure before transport starts causes a consistent fallback. A failure after any P2P work has started fails fast and does not switch to a phase-synchronous path.
