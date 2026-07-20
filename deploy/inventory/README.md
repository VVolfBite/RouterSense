# Deployment inventory

Use `hosts.1x4.example.yaml` for the first single-node run and
`hosts.2x2.example.yaml` for the first multi-node run. Copy one to the
Git-ignored `hosts.local.yaml` and replace placeholders only.

Fields:

- `host`: internal IP used by torchrun rendezvous and NCCL data traffic.
- `ssh_host`: SSH management endpoint; it may equal `host`.
- `port`, `ssh_user`: SSH access from the local controller.
- `current_gpu_count`: GPUs exposed by the instance.
- `target_gpu_count`: GPUs RouterSense will use on that node.
- `remote_rs_root`: disposable remote checkout path.
- `model_cache`: mounted model snapshot directory or parent cache.
- `artifact_root`: remote run/log directory.

Multi-node execution currently requires the same `target_gpu_count` on every
node. Node ranks must be contiguous from zero. The rendezvous port plus one is
reserved for link calibration.
