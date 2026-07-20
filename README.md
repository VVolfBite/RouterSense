# RouterSense

RouterSense is the formal dependency-aware multi-stage communication scheduling
runtime for distributed MoE expert parallelism.

## Active source

- `src/rs/scheduling`: deployable GMWD, RSBC and RSCF planner cores.
- `src/rs/planning`: public planner registry and P012 adapters.
- `src/rs/prediction`: online-safe prediction and FATE contracts.
- `src/rs/runtime/online/megatron_ep`: Megatron observation, planning,
  reconciliation, materialization, execution and evidence.
- `src/rs/runtime/offline`: trace/replay and logical evaluation.
- `src/rs/reference/baselines`: FIFO, Greedy, Birkhoff and related-work style
  references; these are not online planner aliases.
- `scripts/deploy`: local-controller deployment workflow.

Formal dependency direction:

```text
core/topology -> scheduling/planning/prediction -> runtime -> experiments -> deploy
```

`src/rs` must not import experiment or deployment entrypoints. Online runtime
must not use offline references/oracles as a fallback.

## Runtime modes

- Current P012/P0123 planning;
- Future-P012 planning in the previous layer with bounded truth binding;
- Local or Joint scope;
- Event or Global engine;
- GMWD, RSBC or RSCF core;
- phase-sync or asynchronous point-to-point execution.

Predicted P2/P3 is advisory. Executable P0/P1 always comes from actual traffic.
Before transport, canonical task coverage and peer layouts are checked; after
transport, task IDs, rows, bytes and completion evidence are audited.

## Deployment

Read [`DEPLOYMENT_HANDOFF.md`](DEPLOYMENT_HANDOFF.md). The canonical command is:

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-preflight
```

Add `--apply` only after dry-run returns `DRY_RUN_PASS`.

The deployment pipeline distributes a clean local commit, validates the mounted
model and GPU framework, measures directed intra/inter-node GPU communication,
passes one validated cost profile to all planners, launches torchrun, collects
results locally and emits a fail-closed summary.

## Documentation

[`docs/README.md`](docs/README.md) lists the active architecture and evaluation
contracts. Historical round handoffs, recovery notes and retired deployment
commands have been removed from the mainline.

## Validation boundary

CPU/Gloo, trace replay and packaging gates establish source/runtime contracts.
Physical CUDA/NCCL correctness and end-to-end performance require the 1x4 and
2x2 server runs described in the deployment handoff.
