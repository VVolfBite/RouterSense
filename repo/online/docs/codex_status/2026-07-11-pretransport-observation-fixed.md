# 2026-07-11 PreTransport Observation Fixed

## Summary

This round closes the B2 pre-transport traffic-source bug on the CPU/Gloo side.

The formal online planning path no longer derives actual `P0` traffic from early `RuntimeObservation.per_peer_bytes` or any zero/default observation state. The unique runtime planning data chain is now:

1. `PhaseReadyContext`
2. `PreTransportTrafficObservation`
3. local `P0` row from `PhaseReadyContext.send_splits`
4. one gathered `actual_p0_full_row_matrix`
5. prediction audit and next-layer prediction
6. `P1 = transpose(P0)`
7. raw `U`, paired `B`, host projection, safe selection
8. stored selected `P1` plan
9. current `P0` local schedule materialization

## What changed

- Real planning now happens after `build_phase_ready_context(...)`, not before.
- `PreTransportTrafficObservation` is formally defined and captured from `PhaseReadyContext.send_splits` / `recv_splits`.
- `PhaseReadyContext` is now the unique source of actual runtime `P0` traffic for:
  - prediction audit
  - next-layer prediction
  - inferred `P1`
  - raw `U`
  - paired `B`
  - host projection
  - safe selection
  - async local schedule materialization
- `actual_p0_full_row_matrix` is gathered exactly once per `P0` layer.
- `prediction_extra_collective_count` remains `0`.
- `P1` runtime planning still reuses the stored `P0` plan and keeps `p1_planning_collective_count = 0`.
- If dispatcher splits are nonzero but the gathered `P0` matrix is zero, runtime now fail-fast emits `traffic_source_mismatch_rank*.json` instead of falling back.

## Validation

### CPU

- `python -m compileall src experiments tests` passed.
- `PYTHONPATH=src pytest -q tests/contract/test_prepared_window_plan_online.py` passed.
- Added/updated targeted coverage for:
  - `PhaseReadyContext -> PreTransportTrafficObservation`
  - zero-matrix fail-fast with nonzero splits
  - async local materialization still compiling from stored `P0` / inferred `P1`

### Gloo

- Low-level async executor gate passed:
  - `PYTHONPATH=src torchrun --standalone --nproc_per_node=2 experiments/distributed/run_stage1_gloo_e2e_gate.py`
- That confirms under Gloo:
  - nonzero `P0` traffic
  - `batch_isend_irecv`
  - send/recv/local-copy execution
  - two layers and two forward epochs

### Runtime-integrated Gloo

- The runtime-integrated Gloo runner was rewritten to use the formal lifecycle hooks:
  - `before_token_dispatch`
  - `after_token_dispatch`
  - `before_token_combine`
  - `after_token_combine`
- In the current environment, `torchrun --standalone --nproc_per_node=2 experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py` still exits by external `SIGKILL` before result emission.
- This is a remaining Gloo/runtime harness blocker, not a return to the old zero-observation planning path.

## Requested status answers

- real observation capture position:
  - after `build_phase_ready_context(...)`, before any transport starts
- old `RuntimeObservation` removed from planning path:
  - yes
- `PhaseReadyContext` unique data source:
  - yes
- Gloo `actual P0 matrix` nonzero:
  - yes in low-level Gloo gate
- `P1` is exact transpose:
  - enforced and exported in runtime summary
- real `batch_isend_irecv` executed:
  - yes in low-level Gloo gate
- fallback occurred:
  - no in low-level Gloo gate
- `stored/consumed P1` digest equal:
  - enforced in runtime-integrated path and exported; GPU B2 remains untested
- GPU B2 status:
  - `IMPLEMENTED_GPU_UNTESTED`
