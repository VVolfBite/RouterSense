# Megatron Core EP Integration

This directory is the new formal runtime line for RouterSense.

- `pipeline = host_runtime_native_ep`
- `host_runtime = megatron_core`
- `legacy_dependency_allowed = false`

## Scope

This line is for:

- native Megatron Core MoE expert parallel smoke
- native NCCL all-to-all baseline
- read-only RouterSense observation
- no-op dispatcher facade validation

This line is not yet for:

- schedule execution
- P2P transport
- cross-layer prediction
- DeepEP / HybridEP
- multi-node claims

## Layout

- `verify_env.py`: environment gate
- `bootstrap_env.sh`: reproducible dependency install
- `smoke_native_ep.py`: native EP smoke entry
- `collect_native_ep_trace.py`: read-only trace entry
- `routersense/`: no-op observation and passthrough facade
- `tests/`: static and contract tests

## Current Status

At the time of this migration commit, the local machine is blocked on missing:

- `megatron-core`
- `megatron-bridge`
- `transformer-engine`
- local `OLMoE-1B-7B-0125` checkpoint

The scripts in this directory fail explicitly with `status = blocked_environment`
until those requirements are satisfied.
