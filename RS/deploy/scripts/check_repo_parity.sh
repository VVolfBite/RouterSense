#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"

python - "$INVENTORY" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from routesense.topology import inventory_summary, load_inventory

inventory = load_inventory(Path(sys.argv[1]))
summary = inventory_summary(inventory)

def local(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()

def remote(host: dict[str, object], command: str) -> str:
    ssh_host = host.get("ssh_host") or host["host"]
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if host.get("ssh_port") is not None:
        cmd.extend(["-p", str(host["ssh_port"])])
    cmd.append(f'{host["ssh_user"]}@{ssh_host}')
    cmd.append(command)
    return subprocess.check_output(cmd, text=True).strip()

payload = {
    "local_head": local(["git", "rev-parse", "HEAD"]),
    "local_status": local(["git", "status", "--short"]),
    "controller_head": remote(summary["controller"], "cd /root/autodl-tmp/RouterSense && git rev-parse HEAD"),
    "controller_status": remote(summary["controller"], "cd /root/autodl-tmp/RouterSense && git status --short"),
    "executor_head": remote(summary["executor"], "cd /root/autodl-tmp/RouterSense && git rev-parse HEAD"),
    "executor_status": remote(summary["executor"], "cd /root/autodl-tmp/RouterSense && git status --short"),
}
payload["REPO_PARITY_PASS"] = bool(
    payload["local_head"] == payload["controller_head"] == payload["executor_head"]
    and payload["local_status"] == payload["controller_status"] == payload["executor_status"]
)
print(json.dumps(payload, indent=2))
PY

