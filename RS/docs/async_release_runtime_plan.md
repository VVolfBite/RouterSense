# Async Release Runtime Plan

This document tracks the no-GPU AR0/AR1 runtime groundwork.

## Status

- `AR0`: implemented as a fail-closed runtime plan layer.
- `AR1`: experimental P2P executor contract is present.
- Real collectives are still disabled by default.
- `async_release_real_collectives_not_validated=true`.

## AR0

Implemented pieces:

- `src/rs/runtime/online/megatron_ep/async_release/runtime_plan_builder.py`
- unique runtime task ids:
  - `layer:phase:src:dst:bundle:segment:bucket:wave`
- explicit event table:
  - `P0_PLAN_READY`
  - `P0_TRANSFER_COMPLETE`
  - `LOCAL_COMPUTE_COMPLETE`
  - `P1_MATERIALIZED`
  - `P1_TRANSFER_COMPLETE`
  - `P2_FORECAST_READY`
  - `FALLBACK_REQUIRED`
- tensor-compiled schedule contract
- tensor-only agreement helpers

Current guarantee:

- CPU/offline can build an executable-shaped task/event plan from real phase tasks.
- Any mismatch or unsupported condition must fallback to `phase_sync`.

## AR1

Implemented pieces:

- `src/rs/runtime/online/megatron_ep/async_release/p2p_executor.py`
- deterministic peer-op ordering
- recv-before-send ordering
- fake-backend execution report

Current non-claim:

- AR1 does not prove `batch_isend_irecv` correctness on GPU yet.
- It is only an interface/ordering/state-machine scaffold.

## Next GPU Validation

1. gather and validate compiled schedules across ranks with tensor collectives only
2. collect lightweight transport timestamps
3. run AR1 correctness smoke with real ranks
4. keep fallback-to-phase-sync enabled until all-rank ordering is validated
