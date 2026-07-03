# RouteSense Mainline

`RS/` is the formal RouteSense codebase.

## Contents

- `src/rs/`: main runtime, trace, topology, evaluation, and distributed EP code.
- `src/rs/runtime/single_gpu.py`: single-GPU real inference helpers.
- `src/rs/runtime/distributed_ep/core/`: model-agnostic communication and runtime control.
- `src/rs/runtime/distributed_ep/adapter/`: OLMoE-specific EP adapter layer.
- `deploy/`: remote minimal runtime environment and cluster scripts.
- `scripts/`: single-GPU smoke, architecture probe, cluster checks, and deployment helpers.
- `experiments/poc_line1/`: offline scheduler analysis and trace-driven experiments.
- `experiments/distributed/`: distributed bring-up scripts.
- `experiments/ablation/`: reserved configs/scripts for real ablation follow-up.
- `configs/`: project-level model and topology configuration.
- `docs/`: architecture and deployment contracts.
- `tests/`: mainline regression tests.

## Current Phase

The current mainline is being prepared for real distributed OLMoE bring-up. Single-node validation lives in `scripts/`, and distributed EP code is being rebuilt under `src/rs/runtime/distributed_ep/` with a strict `core/adapter` split.

This tree is the only formal RouteSense development path. Legacy POC code remains outside `RS/`.
