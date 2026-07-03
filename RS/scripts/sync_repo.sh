#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
INVENTORY="${1:-$DEFAULT_INVENTORY}"
APPLY=false
FORCE=false
for arg in "$@"; do
  [[ "$arg" == "--apply" ]] && APPLY=true
  [[ "$arg" == "--force" ]] && FORCE=true
done

"$PYTHON_BIN" - "$INVENTORY" "$APPLY" "$FORCE" "$ROOT" <<'PY'
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from rs.topology import inventory_cli_summary, load_inventory

inventory_path = Path(sys.argv[1])
inventory = load_inventory(inventory_path)
apply_mode = sys.argv[2].lower() == "true"
force_mode = sys.argv[3].lower() == "true"
source_root = Path(sys.argv[4]).resolve()
repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=source_root, text=True).strip())
password = os.environ.get("RSSH_PASSWORD") or os.environ.get("SSHPASS")
if not password:
    raise RuntimeError("missing SSH password; set RSSH_PASSWORD or SSHPASS")

payload = {
    "inventory": inventory_cli_summary(inventory, inventory_path=inventory_path),
    "apply_mode": apply_mode,
    "force_mode": force_mode,
    "targets": [],
}
if not apply_mode:
    for node in inventory.nodes:
        payload["targets"].append(
            {
                "node_name": node.name,
                "remote_root": str(node.paths.get("remote_rs_root") or ""),
                "command": f"git bundle create <bundle> HEAD; scp bundle; git fetch/reset on node port {node.port}",
                "status": "DRY_RUN",
            }
        )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0)

local_status = subprocess.check_output(["git", "status", "--short"], cwd=repo_root, text=True).strip()
if local_status and not force_mode:
    raise RuntimeError(f"local tree dirty; refuse to sync without --force: {local_status}")

bundle_dir = Path(tempfile.mkdtemp(prefix="rs-bundle-"))
bundle_path = bundle_dir / "rs.gitbundle"
subprocess.run(["git", "bundle", "create", str(bundle_path), "HEAD"], cwd=repo_root, check=True)
local_ips = {"127.0.0.1", "localhost"}
try:
    local_ips.update(subprocess.check_output(["hostname", "-I"], text=True).split())
except Exception:
    pass
try:
    local_ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
except Exception:
    pass

for node in inventory.nodes:
    remote_root = str(node.paths.get("remote_rs_root") or "")
    if not remote_root:
        raise RuntimeError(f"missing remote_rs_root for {node.name}")
    remote_checkout_root = str(Path(remote_root).parent if Path(remote_root).name == "RS" else Path(remote_root))
    remote_bundle = f"/tmp/rs-{node.name}.gitbundle"
    ssh_host = node.ssh_host or node.host
    if node.host in local_ips or ssh_host in local_ips or Path(remote_checkout_root).resolve() == repo_root:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        payload["targets"].append({"node_name": node.name, "remote_root": remote_root, "checkout_root": remote_checkout_root, "head": head, "status": "APPLIED_LOCAL"})
        continue
    ssh = [
        "sshpass",
        "-p",
        password,
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(node.port),
        f"{node.ssh_user}@{ssh_host}",
    ]
    scp = [
        "sshpass",
        "-p",
        password,
        "scp",
        "-P",
        str(node.port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        str(bundle_path),
        f"{node.ssh_user}@{ssh_host}:{remote_bundle}",
    ]
    subprocess.run(scp, check=True)
    remote_cmd = (
        f"set -euo pipefail; "
        f"mkdir -p {shlex.quote(str(Path(remote_checkout_root).parent))}; "
        f"if [ -d {shlex.quote(remote_checkout_root)}/.git ]; then "
        f"cd {shlex.quote(remote_checkout_root)} && git fetch {shlex.quote(remote_bundle)} HEAD >/dev/null && git reset --hard FETCH_HEAD >/dev/null; "
        f"else "
        f"rm -rf {shlex.quote(remote_checkout_root)} && git clone {shlex.quote(remote_bundle)} {shlex.quote(remote_checkout_root)} >/dev/null && cd {shlex.quote(remote_checkout_root)} && git checkout -f HEAD >/dev/null; "
        f"fi; "
        f"cd {shlex.quote(remote_checkout_root)} && git rev-parse HEAD"
    )
    head = subprocess.check_output(ssh + [remote_cmd], text=True).strip()
    payload["targets"].append({"node_name": node.name, "remote_root": remote_root, "checkout_root": remote_checkout_root, "head": head, "status": "APPLIED"})

print(json.dumps(payload, indent=2))
PY
