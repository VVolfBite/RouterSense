# RouterSense deployment

The canonical operational handoff is [`../DEPLOYMENT_HANDOFF.md`](../DEPLOYMENT_HANDOFF.md).
The local computer is the controller and result sink; PPIO instances are
replaceable workers.

All commands are dry-run unless `--apply` is present. Remote source is never
edited: a clean local commit is transferred through a Git bundle and verified
by commit and canonical tree hash.

## 1. Inventory

```bash
cp deploy/inventory/hosts.1x4.example.yaml deploy/inventory/hosts.local.yaml
# or
cp deploy/inventory/hosts.2x2.example.yaml deploy/inventory/hosts.local.yaml
```

Use the internal PPIO address for `host`. Use `ssh_host` for a separate public
or NAT SSH endpoint. Set `model_cache` to the mounted cloud model directory or
its parent. The model preflight resolves the snapshot and verifies config,
tokenizer, weight index/shards, read permission and local-only loading.

SSH keys are preferred. Password mode is supported through `RSSH_PASSWORD`
when `sshpass` is installed locally.

## 2. Dry-run

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-preflight
```

Expected final status: `DRY_RUN_PASS`.

## 3. Apply

```bash
bash scripts/deploy/run_deployment_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --apply \
  --run-id rs-1x4-smoke
```

Pipeline stages:

1. SSH access and inventory validation;
2. clean Git-bundle source distribution;
3. remote commit/tree parity;
4. Python-side runtime dependency installation;
5. mounted model parity without automatic download, followed by structural/load preflight;
6. CUDA/NCCL/Megatron/Transformer-Engine preflight;
7. directed GPU link calibration and profile distribution;
8. torchrun experiment launch;
9. local result collection;
10. fail-closed result summary.

The GPU image must already provide a CUDA-enabled PyTorch build, Megatron Core,
Megatron Bridge, Transformer Engine, CUDA and NCCL. The pipeline installs the
Python-side packages listed in `deploy/environment/requirements-runtime.txt`
and rejects an incompatible GPU image before calibration or experiment launch.

## Link cost profile

Calibration measures directed rank-pair transfers at multiple token-row sizes,
fits one affine cost per edge, validates it against world size, ranks per node,
model config digest and row-byte size, then supplies the same immutable profile
to Current, Local/Safe and Future planners. Multi-node runtime fails closed if
that profile is missing or mismatched.

## Outputs

Local control report:

```text
outputs/deployment_pipeline/<run-id>/pipeline_report.json
```

Collected remote artifacts and final eligibility decision:

```text
outputs/deployment/<run-id>/
outputs/deployment/<run-id>/deployment_result_summary.json
```

Calibrated topology profile:

```text
outputs/deployment_profiles/<run-id>/link_cost_profile.json
```

## Manual stages

Each stage can be dry-run or applied independently:

```bash
bash scripts/deploy/verify_cluster_access.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/sync_repo.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/verify_repo_parity.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/prepare_cluster_environment.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/sync_model_cache.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/verify_mounted_model.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/verify_runtime_environment.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/calibrate_cluster_links.sh deploy/inventory/hosts.local.yaml --run-id <RUN_ID>
bash scripts/deploy/launch_remote.sh deploy/inventory/hosts.local.yaml --run-id <RUN_ID>
bash scripts/deploy/collect_remote_logs.sh deploy/inventory/hosts.local.yaml --run-id <RUN_ID>
bash scripts/deploy/summarize_collected_run.sh deploy/inventory/hosts.local.yaml --run-id <RUN_ID>
```

## Failure rule

Do not make speculative source/config edits on a worker. Stop the run and return
the pipeline report, stage logs and collected artifacts. CUDA/NCCL performance
is not considered verified until the final collected result summary passes.
