# RouteSense Mainline

`RS/` is the formal RouteSense codebase.

## Contents

- `src/routesense/`: main runtime, trace, topology, evaluation, and distributed EP code.
- `src/routesense/runtime/local_test/`: single-GPU real inference helpers.
- `src/routesense/runtime/distributed_ep/core/`: model-agnostic communication and runtime control.
- `src/routesense/runtime/distributed_ep/adapter/`: OLMoE-specific EP adapter layer.
- `deploy/`: remote minimal runtime environment and cluster scripts.
- `experiments/prerun/`: single-GPU smoke, architecture probe, and router trace checks.
- `experiments/distributed/`: distributed bring-up scripts.
- `experiments/ablation/`: reserved configs/scripts for real ablation follow-up.
- `configs/`: project-level model and topology configuration.
- `docs/`: architecture and deployment contracts.
- `tests/`: mainline regression tests.

## Current Phase

The current mainline is being prepared for real distributed OLMoE bring-up. Pre-run validation lives under `experiments/prerun/`. Distributed EP code is being rebuilt under `runtime/distributed_ep/` with a strict `core/adapter` split.

This tree is the only formal RouteSense development path. Legacy POC code remains outside `RS/`.
