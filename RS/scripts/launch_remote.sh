#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"

python - "$INVENTORY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from routesense.topology import inventory_cli_summary, load_inventory, render_torchrun_dry_run

inventory = load_inventory(Path(sys.argv[1]))
payload = {
    "inventory": inventory_cli_summary(inventory),
    "dry_run": True,
    **render_torchrun_dry_run(inventory, nnodes=2, nproc_per_node=2),
}
payload["gpu_capacity_sufficient"] = all(node.current_gpu_count >= node.target_gpu_count for node in inventory.nodes)
payload["launch_block_status"] = "READY" if payload["gpu_capacity_sufficient"] else "MULTINODE_EP_BLOCKED_BY_GPU_CAPACITY"
print(json.dumps(payload, indent=2))
PY
