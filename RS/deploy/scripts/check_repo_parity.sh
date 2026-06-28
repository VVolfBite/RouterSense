#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"
PASSWORD="${RSSH_PASSWORD:-${SSHPASS:-Helloworld1!}}"

python - "$INVENTORY" <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from routesense.topology import inventory_cli_summary, load_inventory

inventory = load_inventory(Path(sys.argv[1]))
summary = inventory_cli_summary(inventory)
password = os.environ.get("RSSH_PASSWORD") or os.environ.get("SSHPASS") or "Helloworld1!"

def local(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def remote(node: dict[str, object], command: str) -> str:
    cmd = [
        "sshpass",
        "-p",
        password,
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        "-p",
        str(node["port"]),
        f'{node["ssh_user"]}@{node["host"]}',
        command,
    ]
    return subprocess.check_output(cmd, text=True).strip()

def remote_root(node_name: str) -> str:
    value = summary["resolved_paths"].get(f"{node_name}_remote_rs_root")
    if not value:
        raise RuntimeError(f"missing remote root for {node_name}")
    return str(value)

payload = {
    "local_head": local(["git", "rev-parse", "HEAD"]),
    "local_status": local(["git", "status", "--short"]),
    "nodes": [],
}
for node in inventory.nodes:
    head = remote(node.__dict__, f"cd {remote_root(node.name)} && git rev-parse HEAD")
    status = remote(node.__dict__, f"cd {remote_root(node.name)} && git status --short")
    payload["nodes"].append({"name": node.name, "head": head, "status": status})

payload["REPO_PARITY_PASS"] = all(item["head"] == payload["local_head"] and item["status"] == payload["local_status"] for item in payload["nodes"])
print(json.dumps(payload, indent=2))
PY
