#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"

python - "$INVENTORY" <<'PY'
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from routesense.topology import inventory_cli_summary, load_inventory

inventory = load_inventory(Path(sys.argv[1]))
summary = inventory_cli_summary(inventory)
password = os.environ.get("RSSH_PASSWORD") or os.environ.get("SSHPASS")
if not password:
    raise RuntimeError("missing SSH password; set RSSH_PASSWORD or SSHPASS")

def ssh(node, command: str) -> str:
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
        str(getattr(node, "port")),
        f'{getattr(node, "ssh_user")}@{getattr(node, "host")}',
        command,
    ]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()

def remote_root(node_name: str) -> str:
    value = summary["resolved_paths"].get(f"{node_name}_remote_rs_root")
    if not value:
        raise RuntimeError(f"missing remote root for {node_name}")
    return str(value)

payload = {"inventory": summary}
for node in inventory.nodes:
    try:
        payload[f"{node.name}_ssh"] = {
            "ok": True,
            "output": ssh(node, f"cd {remote_root(node.name)} && hostname && python3 -V && git rev-parse HEAD"),
        }
    except subprocess.CalledProcessError as exc:
        payload[f"{node.name}_ssh"] = {"ok": False, "output": exc.output.strip()}

payload["tcp_probe"] = {
    "node_count": len(inventory.nodes),
    "status": "deferred_until_remote_listener_is_enabled",
}
print(json.dumps(payload, indent=2))
PY
