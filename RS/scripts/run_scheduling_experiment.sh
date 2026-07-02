#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"
STRATEGY="${2:-best_of}"

echo "=== Step 1: Check cluster access ==="
bash "$ROOT/scripts/check_cluster_access.sh" "$INVENTORY"

echo
echo "=== Step 2: Verify GPU environment ==="
python3 "$ROOT/scripts/verify_gpu_env.py" || python3 "$ROOT/scripts/verify_cluster_gpu_env.py"

echo
echo "=== Step 3: Sync repository to nodes ==="
bash "$ROOT/scripts/sync_repo.sh" "$INVENTORY" --apply

echo
echo "=== Step 4: Sync model cache ==="
bash "$ROOT/scripts/sync_model_cache.sh" "$INVENTORY" || echo "Model cache sync skipped"

echo
echo "=== Step 5: Render torchrun commands ==="
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 - "$INVENTORY" "$STRATEGY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from routesense.topology import load_inventory

inventory = load_inventory(Path(sys.argv[1]))
strategy = sys.argv[2]
master = next(node for node in inventory.nodes if node.name == inventory.rendezvous.master_node)
commands = {}
nnodes = len(inventory.nodes)
for node in inventory.nodes:
    commands[node.name] = (
        f"ssh -p {node.port} {node.ssh_user}@{node.host} "
        f\"'cd {node.paths.get('remote_rs_root', 'RS')} && "
        f"NCCL_DEBUG=INFO NCCL_SOCKET_TIMEOUT=30 NCCL_IB_DISABLE=0 PYTHONPATH=src torchrun --nnodes={nnodes} --nproc_per_node={node.target_gpu_count} "
        f"--node_rank={node.node_rank} --rdzv-backend={inventory.rendezvous.backend} "
        f"--rdzv-id={inventory.cluster_name}-sched --rdzv-endpoint={master.host}:{inventory.rendezvous.master_port} "
        f"experiments/distributed/exp_scheduled_execution.py --strategy {strategy}'\"
    )

print(json.dumps({
    "inventory": sys.argv[1],
    "strategy": strategy,
    "nnodes": nnodes,
    "commands": commands,
}, indent=2))
PY

echo
echo "=== Done ==="
