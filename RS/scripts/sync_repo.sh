#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"
APPLY=false
FORCE=false
for arg in "$@"; do
  [[ "$arg" == "--apply" ]] && APPLY=true
  [[ "$arg" == "--force" ]] && FORCE=true
done

python - "$INVENTORY" "$APPLY" "$FORCE" "$ROOT" <<'PY'
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from routesense.topology import inventory_cli_summary, load_inventory

inventory = load_inventory(Path(sys.argv[1]))
apply_mode = sys.argv[2].lower() == "true"
force_mode = sys.argv[3].lower() == "true"
source_root = Path(sys.argv[4]).resolve()
password = os.environ.get("RSSH_PASSWORD") or os.environ.get("SSHPASS")
if not password:
    raise RuntimeError("missing SSH password; set RSSH_PASSWORD or SSHPASS")

payload = {"inventory": inventory_cli_summary(inventory), "apply_mode": apply_mode, "force_mode": force_mode, "targets": []}
if not apply_mode:
    for node in inventory.nodes:
        payload["targets"].append(
            {
                "node_name": node.name,
                "remote_root": str(node.paths.get("remote_rs_root") or ""),
                "command": f"git bundle create <bundle> HEAD; rsync bundle; git clone bundle; checkout HEAD on node port {node.port}",
                "status": "DRY_RUN",
            }
        )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0)

local_status = subprocess.check_output(["git", "status", "--short"], cwd=source_root, text=True).strip()
if local_status and not force_mode:
    raise RuntimeError(f"local tree dirty; refuse to sync without --force: {local_status}")

bundle_dir = Path(tempfile.mkdtemp(prefix="rs-bundle-"))
bundle_path = bundle_dir / "routesense.gitbundle"
subprocess.run(["git", "bundle", "create", str(bundle_path), "HEAD"], cwd=source_root, check=True)

for node in inventory.nodes:
    remote_root = str(node.paths.get("remote_rs_root") or "")
    if not remote_root:
        raise RuntimeError(f"missing remote_rs_root for {node.name}")
    remote_bundle = f"{remote_root}/.routesense.gitbundle"
    remote_clone = f"{remote_root}/.routesense.git"
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
        f"{node.ssh_user}@{node.host}",
    ]
    rsync = [
        "sshpass",
        "-p",
        password,
        "rsync",
        "-az",
        "-e",
        f"ssh -p {node.port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
        str(bundle_path),
        f"{node.ssh_user}@{node.host}:{remote_bundle}",
    ]
    subprocess.run(rsync, check=True)
    remote_cmd = (
        f"set -euo pipefail; "
        f"if [ -d {shlex.quote(remote_clone)} ] && [ ! -f {shlex.quote(remote_clone)}/.git ]; then rm -rf {shlex.quote(remote_clone)}; fi; "
        f"rm -rf {shlex.quote(remote_clone)}; "
        f"git clone {shlex.quote(remote_bundle)} {shlex.quote(remote_clone)} >/dev/null; "
        f"cd {shlex.quote(remote_clone)} && git checkout -f HEAD >/dev/null; "
        f"rsync -a --delete --exclude .git {shlex.quote(remote_clone)}/ {shlex.quote(remote_root)}/; "
        f"cd {shlex.quote(remote_root)} && git rev-parse HEAD"
    )
    head = subprocess.check_output(ssh + [remote_cmd], text=True).strip()
    payload["targets"].append({"node_name": node.name, "remote_root": remote_root, "head": head, "status": "APPLIED"})

print(json.dumps(payload, indent=2))
PY
