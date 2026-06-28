# Deploy

Deployment-facing entrypoints live here for RS Phase 0B. The active control-plane flow is two-host and dry-run first.

Recommended checks:

```bash
bash deploy/scripts/check_cluster_access.sh
bash deploy/scripts/check_repo_parity.sh
bash deploy/scripts/launch_remote.sh
```

Single-GPU executor smoke:

```bash
python experiments/deployment/single_gpu_olmoe_smoke.py
python experiments/deployment/single_gpu_router_trace_smoke.py
```

The phase 0B scripts do not claim multi-node expert parallel and do not start NCCL dispatch.
