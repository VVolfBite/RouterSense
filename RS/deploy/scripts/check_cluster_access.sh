#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"
PROBE_PORT="${PROBE_PORT:-29600}"

python - "$INVENTORY" "$PROBE_PORT" <<'PY'
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

from routesense.topology import inventory_summary, load_inventory

inventory = load_inventory(Path(sys.argv[1]))
probe_port = int(sys.argv[2])
summary = inventory_summary(inventory)

def tcp_probe(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False

def ssh_probe(host: dict[str, object]) -> dict[str, object]:
    ssh_host = host.get("ssh_host") or host["host"]
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
    if host.get("ssh_port") is not None:
        cmd.extend(["-p", str(host["ssh_port"])])
    cmd.append(f'{host["ssh_user"]}@{ssh_host}')
    cmd.append("hostname && python3 -V && git rev-parse --short HEAD")
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return {"ok": True, "output": out.strip()}
    except subprocess.CalledProcessError as exc:
        return {"ok": False, "output": exc.output.strip()}

payload = {
    "inventory": summary,
    "controller_ssh": ssh_probe(summary["controller"]),
    "executor_ssh": ssh_probe(summary["executor"]),
    "controller_tcp_to_probe": tcp_probe(summary["controller"].get("ssh_host") or summary["controller"]["host"], probe_port),
    "executor_tcp_to_probe": tcp_probe(summary["executor"].get("ssh_host") or summary["executor"]["host"], probe_port),
}
print(json.dumps(payload, indent=2))
PY

