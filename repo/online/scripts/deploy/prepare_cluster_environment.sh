#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
INVENTORY="${1:-$DEFAULT_INVENTORY}"
APPLY=false
INCLUDE_DEV=false

for arg in "$@"; do
  [[ "$arg" == "--apply" ]] && APPLY=true
  [[ "$arg" == "--include-dev" ]] && INCLUDE_DEV=true
done

"$PYTHON_BIN" - "$INVENTORY" "$APPLY" "$INCLUDE_DEV" "$ROOT" "$PYTHON_BIN" <<'PY'
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

from rs.topology import inventory_cli_summary, load_inventory

inventory_path = Path(sys.argv[1])
apply_mode = sys.argv[2].lower() == "true"
include_dev = sys.argv[3].lower() == "true"
root = Path(sys.argv[4]).resolve()
python_bin = sys.argv[5]
inventory = load_inventory(inventory_path)
summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
password = os.environ.get("RSSH_PASSWORD") or os.environ.get("SSHPASS")

local_ips = {"127.0.0.1", "localhost"}
try:
    local_ips.update(subprocess.check_output(["hostname", "-I"], text=True).split())
except Exception:
    pass
try:
    local_ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
except Exception:
    pass

payload = {
    "inventory": summary,
    "apply_mode": apply_mode,
    "include_dev": include_dev,
    "targets": [],
}

for node in inventory.nodes:
    remote_root = summary["resolved_paths"].get(f"{node.name}_remote_rs_root")
    if not remote_root:
        raise RuntimeError(f"missing remote root for {node.name}")
    ssh_host = node.ssh_host or node.host
    remote_inventory = f"{remote_root}/deploy/inventory/{inventory_path.name}"
    base_cmd = f"cd {shlex.quote(remote_root)} && bash scripts/prepare_node.sh {shlex.quote(remote_inventory)} {shlex.quote(node.name)}"
    if apply_mode:
        base_cmd += " --apply"
    if include_dev:
        base_cmd += " --include-dev"

    if node.host in local_ips or ssh_host in local_ips:
        output = subprocess.check_output(
            ["bash", "-lc", base_cmd],
            text=True,
            cwd=root,
            env=os.environ,
        ).strip()
        payload["targets"].append({"node_name": node.name, "mode": "local", "status": "APPLIED" if apply_mode else "DRY_RUN", "output": output})
        continue

    ssh_cmd = [
        "sshpass",
        "-p",
        password or "",
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(node.port),
        f"{node.ssh_user}@{ssh_host}",
        base_cmd,
    ]
    if ssh_host not in local_ips and not password:
        raise RuntimeError("missing SSH password; set RSSH_PASSWORD or SSHPASS")
    output = subprocess.check_output(ssh_cmd, text=True).strip()
    payload["targets"].append({"node_name": node.name, "mode": "ssh", "status": "APPLIED" if apply_mode else "DRY_RUN", "output": output})

print(json.dumps(payload, indent=2))
PY
