# 2026-07-11 Gloo Gate Progress

## Scope

This handoff captures the current `joint_window_async_p2p` integration state before 4-GPU execution. It is intentionally limited to:

- CPU compile and contract regressions
- real 2-rank Gloo E2E gate
- runtime callgraph reachability
- current known blocker for the remaining stage-1 closure

It does **not** claim:

- 4-GPU `Run B2`
- 4-GPU `Run C2`
- 4-GPU `Run A2`
- faithful FATE validation
- async-release real collective validation on NCCL

## Current Code State

Working tree is intentionally dirty. Main uncommitted areas:

- `joint_window_async_p2p` runtime wiring
- dedicated P2P executor
- lazy-import cleanup for distributed startup stability
- Gloo gate script and tests
- host projection duration fix

Current HEAD during this handoff:

- `83251525a055003e3d84884cc24c100effb50fda`

## What Is Now Real

The following are verified by code path and by real execution:

1. `joint_window_async_p2p` is reachable from the real online runtime configuration path.
2. `execute_async_phase_tensor()` is called from the transport adapter.
3. Real `torch.distributed.batch_isend_irecv()` executes in the 2-rank Gloo gate.
4. Dedicated P2P subgroup creation and warmup succeed.
5. Two consecutive layers and two forward epochs succeed under the Gloo gate.
6. `P0 hidden_states`, `P0 routing_probs`, and `P1 hidden_states` all execute through the async P2P path.
7. No fallback is taken in the Gloo gate.

## Gloo E2E Gate Result

Output directory:

- `outputs/distributed/run_stage1_gloo_e2e_gate`

Verified summary:

- `batch_isend_irecv_executed=true`
- `dedicated_p2p_group_initialized=true`
- `p2p_group_warmup_passed=true`
- `layers_tested=2`
- `forward_epochs_tested=2`
- `per_peer_sequence_validated=true`
- `fallback_used=false`

Important detail:

- Earlier failures were narrowed to low-resource instability, subgroup teardown behavior, and executor/layout bugs.
- With sufficient host resources, the real Gloo gate now completes.

## Tests Run

Successful compile:

```bash
python -m compileall src experiments tests
```

Successful targeted regression set:

```bash
PYTHONPATH=src pytest -q \
  tests/test_experiment_config.py \
  tests/contract/test_prepared_window_plan_online.py \
  tests/contract/megatron_ep/test_async_release_p2p_executor.py \
  tests/contract/megatron_ep/test_runtime_host_projection.py \
  tests/test_architecture_dependencies.py
```

Observed result:

- `60 passed`

Successful real Gloo gate:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 \
  experiments/distributed/run_stage1_gloo_e2e_gate.py
```

## Key Fixes Included In This Handoff

1. Real async P2P executor added:
   - `src/rs/runtime/online/megatron_ep/execution/async_p2p_executor.py`
2. Transport adapter dispatches to async executor in `joint_window_async_p2p`.
3. Receiver offsets now use actual incoming slot layout, not zero default.
4. Gloo teardown stabilized by explicit dedicated subgroup destruction.
5. `p2_matrix` tensor gather path no longer incorrectly falls back under monkeypatched no-default-group tests.
6. Host projection preserves duration when P1 release is delayed by full local P0 completion.
7. Several heavy package `__init__.py` files were converted to lazy import to avoid distributed startup instability.

## What Is Still Not Finished

1. 4-GPU runs are still blocked by hardware visibility:
   - current environment exposes `device_count=1`
2. Full stage-1 closure still requires:
   - `Run B2`
   - `Run C2`
   - `Run A2`
3. Runtime callgraph audit still reports:
   - `host_projection_call.referenced_from_runtime = false`
   This audit output is stale relative to the latest lifecycle change and should be refreshed as part of the next pass.
4. Offline stage-1 unified closure runner and final paper-style comparison tables are not yet complete.

## Practical Next Step

The next highest-value action is **not** more CPU-only interface work. It is:

1. expose 4 visible GPUs
2. rerun the real async runtime on that environment
3. execute:
   - `Run B2`
   - `Run C2`
   - `Run A2`

Until 4 GPUs are visible, the remaining stage-1 runtime claims cannot be closed honestly.

## Explicit Status Flags

- `gpu_not_run_for_4gpu_stage=true`
- `faithful_fate_not_validated=true`
- `async_release_real_collectives_not_validated=true`
- `gloo_e2e_gate_passed=true`
