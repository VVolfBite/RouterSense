# Distributed Experiments

This directory holds deployment-facing experiment entrypoints.

## Naming

- `distributed_*`
  Canonical smoke / bring-up entrypoints for one concrete workflow.
- `exp_*`
  Experiment-oriented entrypoints or compatibility shims kept for operator habit and older docs.

Current compatibility shims:

- `exp_nccl_smoke.py` -> `distributed_nccl_smoke.py`
- `exp_olmoe_ep.py` -> `distributed_olmoe_ep_smoke.py`

## Intended Split

- `runtime/`
  Owns reusable execution framework and model/runtime abstractions.
- `experiments/distributed/`
  Owns invocation shape, CLI defaults, and bring-up sequencing only.

## Current Primary Entry Points

- `distributed_nccl_smoke.py`
  Minimal real NCCL all-reduce / all-gather / all-to-all validation.
- `distributed_olmoe_reference.py`
  Single-GPU reference inference for OLMoE.
- `distributed_olmoe_ep_smoke.py`
  OLMoE EP planning smoke on one visible rank.
- `exp_wave_execution.py`
  Deprecated compatibility shim for legacy distributed trace replay.
- `exp_scheduled_execution.py`
  Scheduler-facing execution experiment harness.
- `future_multinode_smoke.py`
  Dry-run only torchrun contract rendering.

## Real Two-Node Bring-Up

Use `scripts/run_real_cluster_wave.sh` to launch rank 0 remotely and rank 1 locally against a shared inventory.

Example:

```bash
RSSH_PASSWORD='...' \
bash scripts/run_real_cluster_wave.sh \
  deploy/inventory/hosts.ppio.current.yaml \
  U_gated_maxweight_matching_atomic \
  scheduled_collective_partition_replay
```

The script copies the rank-0 JSON result and both logs back under `/tmp/rs_wave_runs/<run-id>/`.

Important boundary:

- `exp_wave_execution.py` is now a deprecated `legacy_trace_replay`
  compatibility path
- it is not the formal online EP benchmark entrypoint
- use `experiments/legacy/exp_trace_replay.py` if this old harness is still
  needed explicitly
- use `experiments/offline/*` and `experiments/online/*` for the formal split
