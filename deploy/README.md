# RouterSense deployment

The active deployment path is inventory-driven and dry-run by default.  No
entrypoint contacts a remote host, installs packages, downloads a model, starts
`torchrun`, stops a process, or copies artifacts unless `--apply` is present.

## 1. Create the private inventory

```bash
cp deploy/inventory/hosts.example.yaml deploy/inventory/hosts.local.yaml
```

Edit both nodes in `hosts.local.yaml`:

- `host`: address reachable by the NCCL/Gloo data plane and rendezvous;
- `ssh_host` (optional): separate SSH management address;
- `ssh_user` and `port`;
- `current_gpu_count` and `target_gpu_count` (formal 2×2 EP uses 2 per node);
- absolute `remote_rs_root`, `model_cache`, and `artifact_root` paths.

`hosts.local.yaml` is git-ignored.  `sync_repo` copies it separately after the
clean Git commit is installed on each node.

SSH keys are used automatically.  Password authentication is also supported:

```bash
export RSSH_PASSWORD='<ssh-password>'
# sshpass must be installed when password authentication is used.
```

For gated Hugging Face models, export `HF_TOKEN` on each target node or in the
remote environment.

## 2. Audit the full pipeline without side effects

```bash
bash scripts/deploy/run_allready_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --run-id rs-dryrun
```

The dry-run covers access, repository sync planning, commit/tree parity,
environment preparation, model synchronization, the two-node `torchrun`
command, and artifact collection.  Its report is written to:

```text
outputs/deployment_pipeline/rs-dryrun/pipeline_report.json
```

## 3. Apply and wait for the deployment smoke

The local source tree must be committed and clean.  The canonical Future-P012
smoke is:

```bash
bash scripts/deploy/run_allready_pipeline.sh \
  deploy/inventory/hosts.local.yaml \
  --apply \
  --run-id rs-future-p012-smoke
```

The default strategy is:

```text
routersense_future_p012_global_rscf_async
```

The default comparison configuration is:

```text
configs/official/online_p012_deploy_smoke.yaml
```

The pipeline waits for both rank groups, records each node's exit code, and
collects the remote run directories into:

```text
outputs/deployment/rs-future-p012-smoke/
```

A different formal strategy can be selected with `--strategy`, for example:

```bash
bash scripts/deploy/run_allready_pipeline.sh deploy/inventory/hosts.local.yaml \
  --apply --run-id rs-p012-local \
  --strategy routersense_p012_local_rscf_async
```

## Individual operations

```bash
# Read-only/dry-run forms
bash scripts/deploy/verify_cluster_access.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/sync_repo.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/verify_repo_parity.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/prepare_cluster_environment.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/sync_model_cache.sh deploy/inventory/hosts.local.yaml
bash scripts/deploy/launch_remote.sh deploy/inventory/hosts.local.yaml

# Side-effecting forms
bash scripts/deploy/verify_cluster_access.sh deploy/inventory/hosts.local.yaml --apply
bash scripts/deploy/sync_repo.sh deploy/inventory/hosts.local.yaml --apply
bash scripts/deploy/verify_repo_parity.sh deploy/inventory/hosts.local.yaml --apply
bash scripts/deploy/prepare_cluster_environment.sh deploy/inventory/hosts.local.yaml --apply
bash scripts/deploy/sync_model_cache.sh deploy/inventory/hosts.local.yaml --apply
bash scripts/deploy/launch_remote.sh deploy/inventory/hosts.local.yaml \
  --apply --wait --run-id rs-manual
bash scripts/deploy/collect_remote_logs.sh deploy/inventory/hosts.local.yaml \
  --apply --run-id rs-manual
bash scripts/deploy/stop_remote_jobs.sh deploy/inventory/hosts.local.yaml \
  --apply --run-id rs-manual
```

## Repository all-ready gate

After extracting the release bundle with its `external_traces/` companion:

```bash
python scripts/verify/run_allready_gate.py --trace-root external_traces
```

The durable JSON and Markdown reports are written under
`outputs/allready/reports/`; every test file also receives an independent log
and timeout boundary.

## Readiness boundary

The repository-level all-ready gate validates source compilation, CPU
contracts, explicit Gloo paths, trace checksums and replay, packaging, and the
complete deployment dry-run.  CUDA/NCCL and physical two-node behavior can only
be marked verified after the commands above run on the target GPU hosts.
