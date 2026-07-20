# Current code structure

The deployable mainline has one direction of dependency:

```text
core/topology -> scheduling/planning/prediction -> runtime -> experiments -> deploy scripts
```

## Active packages

- `src/rs/core`: strict configuration, provenance, metrics, validation.
- `src/rs/topology`: deployment inventory, model-cache resolution, measured
  pairwise link-cost profiles.
- `src/rs/scheduling`: GMWD, RSBC, and RSCF deployable cores plus compilers.
- `src/rs/planning`: public planner registry and P012 adapter.
- `src/rs/prediction`: online-safe predictors and FATE contracts.
- `src/rs/runtime/online/megatron_ep`: observation, planning, bounded repair,
  plan materialization, P2P execution, and evidence.
- `src/rs/runtime/offline`: trace/replay and logical evaluation.
- `src/rs/reference/baselines`: FIFO/Greedy/Birkhoff and related-work style
  references. These are not registered as online planners.
- `src/rs/experiments_support`: reusable experiment implementation.
- `experiments`: thin command-line entrypoints.
- `scripts/deploy`: local-controller deployment pipeline.

## Large-file boundary

`host.py` remains the intentional Megatron/process-group bootstrap boundary.
The remaining large runtime modules are not duplicated alternate implementations;
they are active execution/planning components and are guarded by contract tests.
Cosmetic splitting is deferred until physical CUDA/NCCL validation is complete.

## Retired paths

Historical B/U/Tier1 registries, prediction-policy suites, pending-window runtime,
stage-specific deployment scripts, and old handoff/recovery documents are not
part of the active tree. No formal runtime module may import `experiments` or a
retired compatibility package.
