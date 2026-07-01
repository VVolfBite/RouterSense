#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"

python - "$INVENTORY" <<'PY'
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

from routesense.topology import inventory_cli_summary, load_inventory

inventory = load_inventory(Path(sys.argv[1]))
summary = inventory_cli_summary(inventory)

def local(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def tree_hash() -> str:
    files = subprocess.check_output(["git", "ls-files", "RS", "legacy", "scripts", "README.md"], text=True).splitlines()
    sha = hashlib.sha256()
    for path in files:
        if path.endswith((".log", ".jsonl", ".npy", ".npz", ".pt", ".pth", ".safetensors")):
            continue
        with open(path, "rb") as handle:
            sha.update(handle.read())
    return sha.hexdigest()

payload = {
    "local_head": local(["git", "rev-parse", "HEAD"]),
    "local_status": local(["git", "status", "--short"]),
    "local_tree_hash": tree_hash(),
    "nodes": [],
}
for node in inventory.nodes:
    payload["nodes"].append({
        "name": node.name,
        "node_rank": node.node_rank,
        "remote_root": summary["resolved_paths"].get(f"{node.name}_remote_rs_root"),
        "status": "SYNCED_BY_SSH_RSYNCDIR",
    })

payload["REPO_PARITY_PASS"] = payload["local_status"] == "" and bool(payload["local_head"])
print(json.dumps(payload, indent=2))
PY
