#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"
APPLY=false
for arg in "$@"; do
  [[ "$arg" == "--apply" ]] && APPLY=true
done

python - "$INVENTORY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from routesense.topology import inventory_cli_summary, load_inventory

inventory = load_inventory(Path(sys.argv[1]))
payload = {
    "inventory": inventory_cli_summary(inventory),
    "node_gpu_capacity_ok": all(node.current_gpu_count >= node.target_gpu_count for node in inventory.nodes),
    "status": "READY" if all(node.current_gpu_count >= node.target_gpu_count for node in inventory.nodes) else "MULTINODE_EP_BLOCKED_BY_GPU_CAPACITY",
}
print(json.dumps(payload, indent=2))
PY
