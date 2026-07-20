# RouterSense

RouterSense is a dependency-aware multi-stage communication scheduler and
runtime for distributed MoE expert parallelism. This repository is the
**deployment-precheck mainline**: offline correctness and performance
regressions pass; physical CUDA/NCCL validation must be performed on PPIO.

## Read this first

Codex/PPIO must follow exactly one operational document:

- [`task-test-deploy.md`](task-test-deploy.md)

Do not infer deployment steps from historical reports, archived configs or old
conversation notes. Do not modify Python, planner parameters, experiment YAML or
result JSON on a worker.

Deployment must run from the Git checkout created by the release bundle's
`bootstrap_from_gitbundle.sh`. The standalone source ZIP is an audit/recovery
copy and intentionally contains no `.git`; do not run repo distribution from it.

## First runs

Create the minimal inventory:

```bash
bash scripts/deploy/init_inventory.sh 1x4
# later, after 1x4 passes:
bash scripts/deploy/init_inventory.sh 2x2 --force
```

Only these per-node values normally need editing in
`deploy/inventory/hosts.local.yaml`:

- `host`: PPIO internal/data-plane IP;
- optional `ssh_host`: public/NAT SSH endpoint when different;
- `gpu_count`: GPUs used on that node;
- `model_path`: absolute mounted cloud-storage model path.

`ssh_user` defaults to `root`, SSH port to `22`, remote source/artifact paths,
node names/ranks and rendezvous settings are derived automatically. Override an
optional default only when PPIO reports a different value.

Dry-run:

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-preflight
```

Apply only after the final status is `DRY_RUN_PASS`:

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-1x4-smoke \
  --apply
```

The default experiment config and strategy are already fixed:

- config: `configs/official/online_p012_deploy_smoke.yaml`;
- strategy: `routersense_future_p012_joint_global_rscf_async`.

The pipeline verifies access, distributes a clean Git commit, installs support
packages, verifies the mounted model without downloading, checks CUDA/NCCL and
Megatron dependencies, measures directed GPU links, supplies the measured cost
profile to Current/Local/Safe/Future planners, launches torchrun, collects all
results locally and validates them fail-closed.

## Failure handling

Every run writes:

```text
outputs/deployment_pipeline/<run-id>/pipeline_report.json
outputs/deployment_pipeline/<run-id>/logs/<stage>.log
```

A failed run additionally writes:

```text
outputs/deployment_pipeline/<run-id>/failure_summary.txt
```

The failure summary contains the failed stage, command, return code, timeout
state, local log tail and any collected remote log tails. Codex must stop,
return these artifacts and avoid speculative source/config edits.

## Active code

- `src/rs/scheduling`: deployable GMWD, RSBC and RSCF cores;
- `src/rs/planning`: planner registry and P012 adapters;
- `src/rs/prediction`: FATE/prediction contracts;
- `src/rs/runtime/online/megatron_ep`: observation, prediction, reconciliation,
  materialization, execution and evidence;
- `src/rs/runtime/offline`: replay and logical evaluation;
- `src/rs/reference/baselines`: FIFO, Greedy, Birkhoff and related-work-style
  references, never online fallback aliases;
- `scripts/deploy`: local-controller deployment pipeline.

Predicted future traffic is advisory. Executable P0/P1 is rebuilt from truth.
Before transport, canonical task coverage and peer layouts are checked; after
transport, task IDs, row/byte counts and completion evidence are audited.
Prediction reconciliation removes obsolete edges, resizes retained edges and
inserts each actual edge once; new P1 work remains blocked until release.

## Preserved offline evidence

The prior same-condition FAST-style comparison is preserved at
[`docs/results/joint_vs_fast_ep_record_20260720.md`](docs/results/joint_vs_fast_ep_record_20260720.md).
Across 252 windows, RSCF Joint reduced logical communication makespan relative
to FAST-style by 9.79%, 13.45%, 16.19% and 19.35% at EP=4, 8, 12 and 16.
This is an offline logical-time record, not a hardware performance claim.

## Validation boundary

The current source passed the repository CPU/Gloo, trace and offline regression
gates before this documentation/configuration handoff. PPIO must still validate:

1. 1 node × 4 GPUs: environment, mounted model, calibration, tensor replacement
   and basic performance;
2. 2 nodes × 2 GPUs: internal networking, NCCL, cross-node calibration and
   distributed result parity.
