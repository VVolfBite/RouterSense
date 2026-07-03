# Deploy

Deployment-facing entrypoints live here for the current RS bring-up path. The active control-plane flow is inventory-driven, dry-run first, then NCCL / OLMoE execution smokes.

Recommended checks:

```bash
bash scripts/check_cluster_access.sh
bash scripts/check_repo_parity.sh
bash scripts/launch_remote.sh
```

Single-GPU executor smoke:

```bash
python scripts/single_gpu_smoke.py
python scripts/router_trace_smoke.py
```

These helpers are deployment-oriented wrappers. Real multi-rank validation happens in `experiments/distributed/`.
