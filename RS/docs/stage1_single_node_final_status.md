# Stage1 Single-Node Final Status

Date: `2026-07-11`

Current status is "final pre-GPU closure". The code path is prepared for the next step:

`4GPU Run B2 -> Run C2 -> Run A2`

This round did not execute 4GPU NCCL runs because the current environment exposes only one CUDA device. The remaining work is GPU execution, not more CPU/runtime feature expansion.

## What Is Now Real In Runtime

- `joint_window_async_p2p` is reachable from the real Megatron host hook chain.
- The async path reaches:
  - runtime lifecycle
  - joint-window planning
  - host-projected safe U/B selection
  - local async schedule materialization
  - `torch.distributed.batch_isend_irecv`
- `P1` reuses the `P0`-stored plan.
- `P1 planning collective count = 0`.
- `prediction extra collective count = 0`.
- dedicated P2P groups are created in globally consistent order during initialization, not in the hot path.
- task coalescing is disabled for first-stage GPU correctness.

## Current Guarantees

- `real_async_executor_reachable=true`
- `dedicated_group_order_validated=true`
- `coalescing_enabled=false`
- `per_peer_sequence_validated=true` in debug/execution Gloo gates
- `microbatch_sequence_isolated=true`
- `preflight_validated=true`
- `post_start_fallback_forbidden=true`
- `host_projected_safe_selection_active=true`
- `birkhoff_same_executor_ready=true`
- `runtime_integrated_gloo_passed=true`

## Current Limits

- GPU environment available now: `torch.cuda.device_count() == 1`
- 4GPU `Run B2 / C2 / A2` are prepared but not executable in this environment
- faithful FATE is still not implemented
- async P2P real NCCL path is prepared but not yet validated on 4GPU NCCL
- no per-bucket or per-expert compute overlap is claimed

## Direct Answers

- Real P2 is generated from current-layer actual P0 summary and stored during `P0`.
- The prediction is consumed during current-layer joint-window planning and guides current `P1` as future pressure for `P0(L+1)`.
- `P1` does not rebuild prediction, U/B, or global plan.
- Each async layer target is:
  - `P0`: one compact global summary gather, one digest agreement
  - `P1`: zero planning gather, zero full-plan broadcast
- Safe fallback to `B` still uses the same async executor.

## Ready Next Step

1. Run `Run B2` on real 4GPU NCCL and confirm prediction lifecycle, preflight, sequence validation, and zero extra collectives.
2. Run `Run C2` and require real async P2P parity with no fallback.
3. Run `Run A2` with async fairness baselines:
   - `fifo_async_p2p`
   - `greedy_async_p2p`
   - `birkhoff_phase_local_async_p2p`
   - `routersense_joint_zero_hint_async_p2p`
   - `routersense_joint_predicted_async_p2p`

## Current Remaining Risk

The remaining risk is no longer CPU integration. It is 4GPU NCCL runtime validation:

- subgroup creation order on real NCCL
- real sequence parity across two consecutive layers and two forward epochs
- real async transport parity for `hidden_states` and `routing_probs`
- whether joint predicted async beats async Birkhoff once transport and control are held fixed
