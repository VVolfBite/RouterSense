# Deployment inventory

Create the local inventory with:

```bash
bash scripts/deploy/init_inventory.sh 1x4
# or, after 1x4 passes:
bash scripts/deploy/init_inventory.sh 2x2 --force
```

The compact schema derives stable RouterSense settings automatically. Normally
only these values are required per node:

- `host`: internal IP used by torchrun/NCCL;
- optional `ssh_host`: management endpoint when different;
- `gpu_count`: GPUs used by RouterSense on that node;
- `model_path`: absolute mounted model directory or parent cache.

Optional overrides:

- `ssh_user` when not `root`;
- `ssh_port` when not `22`;
- `remote_rs_root` and `artifact_root` only when the PPIO image forbids the
  defaults `/workspace/RouterSense` and `/workspace/routersense-artifacts`.

Node names, ranks, current/target GPU counts, rendezvous master, port and backend
are inferred. Multi-node execution requires the same `gpu_count` on every node.
