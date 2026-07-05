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

## RouterSense Integration Rules

- `observer.py` is a collection point only.
  It stores raw or near-raw runtime metadata and must not participate in execution decisions.
- Any observer failure must degrade to warnings or missing rows, not runtime failure.
- `dispatcher_facade.py` is the scheduling injection seam.
  In the current stage it is strict no-op passthrough and only records injected config.
- Future scheduling work should enter through the facade, not through observer-side processing.

## Current Status

The local environment now imports and executes:

- `megatron-core`
- `megatron-bridge`
- `transformer-engine`
- `nvidia-modelopt`

Validated locally on:

- single node
- 2 visible GPUs
- NCCL backend
- OLMoE-1B-7B-0924 local checkpoint
- EP=2, dispatcher=`alltoall`

Validated gates:

- Gate A: native EP=2 NCCL forward with real remote dispatch/combine
- Gate B: lightweight observer trace export for P0/P1
- Gate C: no-op `native_order` facade numerical equivalence

Current default model path:

- `/root/autodl-tmp/models/OLMoE-1B-7B-0924`

Remaining practical constraints are:

- no scheduled transport policy is implemented yet
- observer/facade remain read-only / no-op by design in this stage

`verify_env.py` distinguishes between:

- environment blocked (`status = blocked_environment`)
- environment ready (`status = ready`)

and the smoke / trace scripts now attempt real native Megatron EP execution
once the environment is ready.
