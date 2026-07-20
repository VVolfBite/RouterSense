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

## Planning modes

The online runtime preserves the original P012 path and exposes two isolated
extensions through the same formal planner/runtime chain:

- `routersense_p012_async`: on-demand P012.  Current P0/P1 are executable and
  predicted P2 is advisory.
- `routersense_p0123_async`: on-demand P0123 advisory horizon.  The planner also
  considers `P3 = transpose(P2)`, but executable coverage remains current P0/P1.
- `routersense_future_p012_async`: the unchanged P012 planner runs in the
  previous layer and publishes an immutable target-layer plan.  The target
  layer binds actual P0/P1 and performs at most the existing bounded repair;
  it does not run a second full planner.

See `docs/P012_P0123_FUTURE_P012.md` and
`configs/comparison/p012_p0123_future_p012.yaml`.

## Formal exact Oracle

The current paper `O_local`/`O_joint` comparison uses the same certified tiny
canonical bucket-wave model and changes only scheduling scope:

```text
O_local: exact P0/P1/P2 phase-local solve
O_joint: exact rank-release-aware joint solve
```

The model ID is `routersense_exact_bucket_wave_release_v2`. It is limited to at
most 4 ranks and 12 canonical bucket tasks and fails closed above that scale.
Historical CP-SAT names are compatibility aliases; Birkhoff fluid decomposition
is a separate sensitivity reference. See
`reports/UNIFIED_ORACLE_MODEL_20260719.md`.

## Deployment entry

Create a private inventory from `deploy/inventory/hosts.example.yaml`, then run:

```bash
bash scripts/deploy/run_allready_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id <RUN_ID>
```

The default deployment comparison is defined in:

```text
configs/official/online_p012_deploy_smoke.yaml
```

Before the asynchronous target-layer planner accepts work, runtime attach
performs a process-once cold-start warmup for the production Future-P012 Event
and Global planners, ordinary P2 truth binding, and prepared-order binding.
Machine-specific Python and Numba caches are excluded from source archives.

Run the recoverable code/deployment gate with an external trace bundle:

```bash
python scripts/verify/run_allready_gate.py \
  --trace-root /path/to/routersense_traces \
  --output-dir outputs/allready
```

For a source-only host without the external trace bundle, use
`--skip-trace-modes`. The report marks both unavailable trace stages as
`SKIPPED`; it does not treat them as newly reproduced evidence.
