# 2026-07-11 Final Pre-GPU Closure

Base during this round:

- dirty tree continued from prior handoff
- no reset / no discard
- current environment GPU visibility: `device_count=1`

## Completed

- dedicated P2P subgroup creation now uses global identical `new_group` ordering
- hot path `new_group` count stays zero
- task coalescing is disabled for first-stage GPU correctness
- async preflight validates send/recv rows, coverage, and sequence parity before first P2P
- `use_nccl_stream` is explicitly passed through transport adapter
- runtime joint planning now performs host-projected safe U/B selection and stores the selected plan
- runtime-integrated Gloo gate now reaches:
  - prediction
  - runtime joint plan
  - stored `P1` plan reuse
  - transport adapter
  - `batch_isend_irecv`
- same-executor async fairness baselines are configured
- perf path suppresses timeline and heartbeat hot-path recording

## Verified In This Environment

- targeted pytest: `59 passed`
- low-level Gloo gate: passed
- runtime-integrated Gloo gate: passed
- `git diff --check`: passed

## Key Runtime Facts

- `joint_window_async_p2p` is host-reachable
- dedicated P2P groups are initialized once
- `batch_isend_irecv_call_count > 0` in runtime-integrated Gloo gate
- safe-selected policy is non-empty in runtime-integrated gate
- `prediction_extra_collective_count = 0`
- `p1_planning_collective_count = 0`
- fallback count remained `0` in both Gloo gates

## Still Not Executed

- 4GPU `Run B2`
- 4GPU `Run C2`
- 4GPU `Run A2`

Reason:

- current host exposes only one CUDA device

## Status Flags

- `real_async_executor_reachable=true`
- `dedicated_group_order_validated=true`
- `coalescing_enabled=false`
- `per_peer_sequence_validated=true`
- `microbatch_sequence_isolated=true`
- `preflight_validated=true`
- `post_start_fallback_forbidden=true`
- `host_projected_safe_selection_active=true`
- `birkhoff_same_executor_ready=true`
- `p1_planning_collective_count=0`
- `prediction_extra_collective_count=0`
- `perf_hotpath_json_count=0` for the perf-specific code paths touched this round
- `perf_hotpath_filesystem_write_count=0` for the perf-specific code paths touched this round
- `runtime_integrated_gloo_passed=true`
- `gpu_run_b2_ready=true`
- `gpu_run_c2_ready=true`
- `gpu_run_a2_ready=true`

## Next Step

Do not add more CPU/runtime features before GPU.

Next step is exactly:

`Run B2 -> Run C2 -> Run A2 -> one focused GPU-side optimization pass if needed`
