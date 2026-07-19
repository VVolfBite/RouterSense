#!/usr/bin/env python3
from __future__ import annotations

"""Synchronize the clean local RouterSense commit and private inventory to nodes."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.topology import inventory_cli_summary, load_inventory
from scripts.deploy.launch_remote import _local_addresses, _run_node


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow replacing an existing remote checkout; local source must still be clean",
    )
    return parser.parse_args(argv)


def _auth_prefix() -> list[str]:
    password = os.environ.get("RSSH_PASSWORD") or os.environ.get("SSHPASS")
    if not password:
        return []
    if shutil.which("sshpass") is None:
        raise RuntimeError("RSSH_PASSWORD/SSHPASS is set but sshpass is not installed")
    return ["sshpass", "-p", password]


def _copy_to_node(node: Any, source: Path, target: str, *, local_addresses: set[str]) -> None:
    host = str(node.ssh_host or node.host)
    if str(node.host) in local_addresses or host in local_addresses:
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    command = [
        *_auth_prefix(),
        "scp",
        "-P",
        str(node.port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        str(source),
        f"{node.ssh_user}@{host}:{target}",
    ]
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory).resolve()
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip())
    local_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    local_status = subprocess.check_output(["git", "status", "--short"], cwd=repo_root, text=True).strip()
    payload: dict[str, Any] = {
        "schema_version": "routersense.deploy.repo_sync.v2",
        "inventory": summary,
        "apply_mode": bool(args.apply),
        "force_mode": bool(args.force),
        "local_head": local_head,
        "local_clean": not bool(local_status),
        "targets": [],
    }
    for node in inventory.nodes:
        remote_root = str(summary["resolved_paths"].get(f"{node.name}_remote_rs_root") or "")
        if not remote_root:
            raise RuntimeError(f"missing remote_rs_root for {node.name}")
        payload["targets"].append(
            {
                "node_name": node.name,
                "remote_root": remote_root,
                "inventory_target": f"{remote_root}/deploy/inventory/{inventory_path.name}",
                "command": "git bundle transfer + hard reset + private inventory copy",
                "status": "DRY_RUN",
            }
        )
    if not args.apply:
        payload["status"] = "DRY_RUN"
        print(json.dumps(payload, indent=2))
        return 0
    if local_status:
        payload["status"] = "FAIL"
        payload["reason"] = "local tree is dirty; commit or clean changes before deployment sync"
        payload["local_status"] = local_status
        print(json.dumps(payload, indent=2))
        return 2

    local_addresses = _local_addresses()
    with tempfile.TemporaryDirectory(prefix="rs-bundle-") as temp_dir:
        bundle_path = Path(temp_dir) / "rs.gitbundle"
        subprocess.run(["git", "bundle", "create", str(bundle_path), "HEAD"], cwd=repo_root, check=True)
        applied: list[dict[str, Any]] = []
        for node in inventory.nodes:
            remote_root = str(summary["resolved_paths"][f"{node.name}_remote_rs_root"])
            remote_bundle = f"/tmp/rs-{node.name}-{local_head[:12]}.gitbundle"
            host = str(node.ssh_host or node.host)
            local_mode = str(node.host) in local_addresses or host in local_addresses
            if local_mode and Path(remote_root).resolve() == repo_root.resolve():
                inventory_target = repo_root / "deploy" / "inventory" / inventory_path.name
                if inventory_target.resolve() != inventory_path:
                    inventory_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(inventory_path, inventory_target)
                applied.append(
                    {
                        "node_name": node.name,
                        "remote_root": remote_root,
                        "head": local_head,
                        "mode": "local_existing",
                        "inventory_target": str(inventory_target),
                        "status": "PASS",
                    }
                )
                continue

            if local_mode:
                remote_bundle_path = Path(remote_bundle)
                shutil.copy2(bundle_path, remote_bundle_path)
            else:
                _copy_to_node(node, bundle_path, remote_bundle, local_addresses=local_addresses)
            remote_cmd = (
                "set -euo pipefail; "
                f"mkdir -p {shlex.quote(str(Path(remote_root).parent))}; "
                f"if [ -d {shlex.quote(remote_root + '/.git')} ]; then "
                f"cd {shlex.quote(remote_root)}; "
                f"git fetch {shlex.quote(remote_bundle)} HEAD >/dev/null; "
                "git reset --hard FETCH_HEAD >/dev/null; git clean -fdx -e deploy/inventory/*.local.yaml >/dev/null; "
                "else "
                + (f"rm -rf {shlex.quote(remote_root)}; " if args.force else "")
                + f"git clone {shlex.quote(remote_bundle)} {shlex.quote(remote_root)} >/dev/null; "
                f"fi; git -C {shlex.quote(remote_root)} rev-parse HEAD"
            )
            head = _run_node(node, remote_cmd, local_addresses=local_addresses).splitlines()[-1]
            inventory_target = f"{remote_root}/deploy/inventory/{inventory_path.name}"
            prep_inventory = f"mkdir -p {shlex.quote(str(Path(inventory_target).parent))}"
            _run_node(node, prep_inventory, local_addresses=local_addresses)
            _copy_to_node(node, inventory_path, inventory_target, local_addresses=local_addresses)
            _run_node(node, f"rm -f {shlex.quote(remote_bundle)}", local_addresses=local_addresses)
            applied.append(
                {
                    "node_name": node.name,
                    "remote_root": remote_root,
                    "head": head,
                    "mode": "local" if local_mode else "ssh",
                    "inventory_target": inventory_target,
                    "status": "PASS" if head == local_head else "FAIL",
                }
            )
        payload["targets"] = applied
    payload["status"] = "PASS" if all(row["status"] == "PASS" for row in payload["targets"]) else "FAIL"
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
