#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
INVENTORY="${1:-$DEFAULT_INVENTORY}"
NODE_NAME="${2:-node0}"

"$PYTHON_BIN" - "$INVENTORY" "$NODE_NAME" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from rs.topology import load_inventory, resolve_node_model_cache, resolve_node_rs_root

inventory = load_inventory(Path(sys.argv[1]))
node_name = sys.argv[2]
node = next(node for node in inventory.nodes if node.name == node_name)
payload = {
    "node_name": node.name,
    "model_id": "allenai/OLMoE-1B-7B-0924-Instruct",
    "revision": "unknown",
    "cache_path": str(resolve_node_model_cache(inventory, node_name) or ""),
    "remote_rs_root": str(resolve_node_rs_root(inventory, node_name) or ""),
    "total_size_bytes": 0,
    "required_files_present": False,
    "tokenizer_ready": False,
    "config_ready": False,
    "weights_ready": False,
    "gpu_runtime_attempted": False,
}
print(json.dumps(payload, indent=2))
PY
