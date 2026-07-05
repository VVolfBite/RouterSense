# Deploy

Deployment-facing entrypoints live here for the formal RS bring-up path. The
active workflow is inventory-driven, dry-run first, then explicit experiment
launch via the formal online entrypoints.

Recommended checks:

```bash
bash scripts/deploy/verify_cluster_access.sh
bash scripts/deploy/verify_repo_parity.sh
bash scripts/deploy/launch_remote.sh
```

Recommended environment preparation after repo sync:

```bash
export RSSH_PASSWORD='<ssh-password>'
bash scripts/deploy/sync_repo.sh deploy/inventory/hosts.local.yaml --apply --force
bash scripts/deploy/prepare_cluster_environment.sh deploy/inventory/hosts.local.yaml --apply
```

Single-GPU executor smoke:

```bash
python scripts/diagnostics/run_single_gpu_smoke.py
python scripts/diagnostics/run_router_trace_smoke.py
```

Formal multi-rank validation is launched through `experiments/online/` and the
remote wrapper in `deploy/remote/`.
