# RouteSense Mainline

`RS/` is the formal RouteSense codebase.

## Contents

- `src/rs/core/`: shared contracts, artifact helpers, hashing, validation.
- `src/rs/scheduling/`: formal scheduling ABI and policy surface.
- `src/rs/runtime/offline/`: formal offline trace, traffic, prediction, and runner surface.
- `src/rs/runtime/online/megatron_ep/`: formal online Megatron EP runtime surface.
- `experiments/offline/`: formal offline experiment entrypoints.
- `experiments/online/`: formal online experiment entrypoints.
- `deploy/`: remote minimal runtime environment and cluster templates.
- `scripts/`: deployment, verification, plotting, and maintenance helpers.
- `configs/`: model, topology, workload, and experiment configuration.
- `docs/`: architecture, handoff, experiment, and operations documentation.
- `archive/`: small milestone manifests and fingerprints only.
- `artifacts/`: raw run outputs, not tracked by Git.
- `tests/`: mainline regression tests.

## Formal Dependency Direction

```text
core
  ↑
scheduling
  ↑
runtime/offline        runtime/online
  ↑                        ↑
experiments
  ↑
scripts / deploy
```

Forbidden reverse dependencies:

- `scheduling` must not import torch, Megatron, NCCL, experiment runners, or artifact paths.
- `runtime` must not import `experiments`.
- `experiments` must not implement reusable runtime logic.
- `legacy` must not be imported by formal runtime code.

## Current Formal Mainline

The formal online runtime path is now:

```text
src/rs/runtime/online/megatron_ep/
```

The formal offline runtime path is now:

```text
src/rs/runtime/offline/
```

The formal scheduling path is now:

```text
src/rs/scheduling/
```

The formal mainline now routes all supported runtime, offline, and policy work
through these packages. Historical material is parked under `legacy/` and is
excluded from the default validation path.
