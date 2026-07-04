# Current PPIO Bring-Up Notes

This note captures the current two-node test target as of 2026-07-03.

## Nodes

- `node0` remote PPIO worker
  - private IP: `172.20.139.12`
  - SSH entry: `root@proxy.cn-south-1.gpu-instance.ppinfra.com -p 47769`
  - RouteSense root: `/vllm-workspace/RouterSense/RS`
  - model cache root: `/vllm-workspace/models`
- `node1` local worker
  - private IP: `172.20.114.142`
  - RouteSense root: `/root/RouterSense/RS`
  - model cache root: `/root/model-cache`

## Important Constraints

- The repository's older `deploy/inventory/hosts.local.yaml` points to a previous SeeTaCloud setup and is not valid for this run.
- The local machine currently has no `sshd`, so `launch_remote.sh` style automation cannot control both nodes symmetrically.
- The current practical bring-up path is manual two-terminal `torchrun`, with `node0` as rendezvous master.
- Both machines currently expose one visible GPU, so the immediate realistic target is `2 nodes x 1 GPU`, not `2 nodes x 2 GPU`.

## Inventory

Prepared inventory for this pair:

- `deploy/inventory/hosts.ppio.current.yaml`

This file is intended for path resolution and dry-run reference. It should not be treated as proof that local SSH automation is ready.

## Primary Test Order

1. Confirm repo present on both nodes.
2. Confirm model cache present on both nodes.
3. Run `distributed_nccl_smoke.py` under `torchrun` across 2 ranks.
4. Run `exp_wave_execution.py` in `unscheduled_collective_replay`.
5. Run `exp_wave_execution.py` in `wave_collective_replay` or `scheduled_collective_partition_replay`.

## Manual Torchrun Skeleton

Remote `node0`:

```bash
cd /vllm-workspace/RouterSense/RS
PYTHONPATH=src NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --rdzv-backend=c10d --rdzv-id=rs-ppio-current \
  --rdzv-endpoint=172.20.139.12:29500 \
  experiments/distributed/distributed_nccl_smoke.py
```

Local `node1`:

```bash
cd /root/RouterSense/RS
PYTHONPATH=src NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 \
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --rdzv-backend=c10d --rdzv-id=rs-ppio-current \
  --rdzv-endpoint=172.20.139.12:29500 \
  experiments/distributed/distributed_nccl_smoke.py
```

If the NCCL smoke passes, swap the script path to:

```bash
experiments/distributed/exp_wave_execution.py \
  --model-path <node-local-model-path> \
  --runtime-mode trace_replay \
  --execution-mode unscheduled_collective_replay
```

and then:

```bash
experiments/distributed/exp_wave_execution.py \
  --model-path <node-local-model-path> \
  --runtime-mode trace_replay \
  --execution-mode wave_collective_replay \
  --strategy U_gated_maxweight_matching
```

## Current Known State

- Remote model download completed at `/vllm-workspace/models/OLMoE-1B-7B-0924-Instruct`
- Local model download is still incomplete and may need either a retry or a remote-to-local copy

## Scope Boundary

These bring-up steps currently validate only `trace_replay` distributed wiring.
They do not validate a real EP runtime, online prediction, or production EP
performance.
