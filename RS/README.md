# RouteSense Mainline

`RS/` is the formal RouteSense codebase.

## Contents

- `src/rs/core/`: shared contracts, artifact helpers, hashing, validation.
- `src/rs/scheduling/`: formal scheduling ABI and policy surface.
- `src/rs/runtime/offline/`: formal offline trace, traffic, prediction, and runner surface.
- `src/rs/runtime/online/megatron_ep/`: formal online Megatron EP runtime surface.
- `src/rs/legacy/`: deprecated compatibility shims and legacy metadata.
- `experiments/offline/`: formal offline experiment entrypoints.
- `experiments/online/`: formal online experiment entrypoints.
- `experiments/poc_line1/`: retained historical offline study entrypoints.
- `experiments/distributed/`: retained historical bring-up scripts for legacy paths.
- `deploy/`: remote minimal runtime environment and cluster templates.
- `scripts/`: deployment, verification, plotting, and maintenance helpers.
- `configs/`: project-level model, topology, placement, workload, and scheduler configuration.
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

Round-1 cleanup keeps the already-validated implementations in place and exposes
them under these new package paths through compatibility wrappers. This avoids
algorithm or executor behavior changes while letting future work depend on a
single formal import surface.

## Legacy Boundary

The following code remains in-tree only as historical reference and must not be
used as the formal runtime path:

- `src/rs/online/olmoe_ep/`
- `src/rs/runtime/distributed_ep/`
- `experiments/distributed/`
- `experiments/online/bench_native_ep.py`
- `experiments/online/bench_scheduled_ep.py`

See `legacy/hf_olmoe_ep_harness/README.md` for the policy boundary.
