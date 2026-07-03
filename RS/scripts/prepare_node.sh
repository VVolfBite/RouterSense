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

from rs.topology import inventory_cli_summary, load_inventory, resolve_node_artifact_root, resolve_node_model_cache, resolve_node_rs_root

inventory_path = Path(sys.argv[1])
inventory = load_inventory(inventory_path)
node_name = sys.argv[2]
payload = {
    "node_name": node_name,
    "inventory": inventory_cli_summary(inventory, inventory_path=inventory_path),
    "remote_rs_root": str(resolve_node_rs_root(inventory, node_name) or ""),
    "model_cache": str(resolve_node_model_cache(inventory, node_name) or ""),
    "artifact_root": str(resolve_node_artifact_root(inventory, node_name) or ""),
    "required_files_present": False,
    "tokenizer_ready": False,
    "config_ready": False,
    "weights_ready": False,
    "gpu_runtime_attempted": False,
}
print(json.dumps(payload, indent=2))
PY
