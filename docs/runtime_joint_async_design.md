# Runtime Joint Async Design

## Final Stage1 Runtime Path

The final single-node runtime path is:

`Megatron host hook -> RouterSense runtime lifecycle -> joint window plan -> host-projected safe selection -> async local schedule materialization -> batch_isend_irecv`

Execution mode:

- `joint_window_async_p2p`

Main public strategy names prepared for the same executor:

- `fifo_async_p2p`
- `greedy_async_p2p`
- `birkhoff_phase_local_async_p2p`
- `routersense_joint_zero_hint_async_p2p`
- `routersense_joint_predicted_async_p2p`
- `routersense_safe_joint_async`

## Planning Semantics

For each layer `L` during `P0`:

1. Build local `PhaseReadyContext`.
2. Perform one compact global summary gather.
3. Reconstruct the global actual `P0(L)` matrix.
4. Audit any previous prediction targeting `L`.
5. Generate prediction for `P0(L+1)`.
6. Infer `P1(L)` from `P0(L)`.
7. Build one runtime joint problem:
   - actual `P0(L)`
   - inferred `P1(L)`
   - predicted `P0(L+1)`
8. Build raw `U`.
9. Build paired `B`.
10. Apply the same host projection to both.
11. Safe-select `U` or `B`.
12. Materialize local async `P0` schedule.
13. Store abstract `P1` plan for reuse at `before_token_combine`.

`P1` does not:

- rebuild prediction
- rebuild U/B
- re-run safe selection
- re-gather traffic
- broadcast a full plan

## Host-Projected Safe Selection

The online runtime no longer executes raw lookahead blindly.

It now compares:

- host-projected raw `U`
- host-projected paired `B`

The active selected plan is the projected-safe winner, not the ideal replay winner.

## Transport Semantics

Stage1 only claims rank-level asynchronous phase completion:

- `P0_LOCAL_PHASE_COMPLETE`
  - both roles complete:
    - `hidden_states`
    - `routing_probs`
  - then `token_dispatch()` may return
- `P1_LOCAL_PHASE_COMPLETE`
  - rank-local `P1` receives complete
  - then `token_combine()` may continue

Stage1 does **not** claim:

- per-bucket expert compute overlap
- per-expert release
- cancellation after P2P work has started

## Safety Rules

Before first P2P work:

- validate global plan digest
- validate sequence parity
- validate send/recv rows
- validate dtype/shape suffix parity
- validate receive coverage
- validate dedicated P2P group initialization

If preflight fails before first work:

- all ranks fall back consistently before P2P starts

If failure happens after any work has started:

- fail fast
- no phase-sync fallback

## Dedicated Communicator

Control collectives and async P2P data path do not share hot-path group creation.

All world ranks create all unique EP subgroup P2P groups in the same sorted order at initialization time.
Each runtime only keeps the handle for its own subgroup.
