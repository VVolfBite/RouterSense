#!/usr/bin/env python3
from __future__ import annotations

"""Collect one deployment run directory from every inventory node."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.topology import inventory_cli_summary, load_inventory
from scripts.deploy.launch_remote import _local_addresses


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def _scp_base() -> list[str]:
    password = os.environ.get("RSSH_PASSWORD") or os.environ.get("SSHPASS")
    base: list[str] = []
    if password:
        if shutil.which("sshpass") is None:
            raise RuntimeError("RSSH_PASSWORD/SSHPASS is set but sshpass is not installed")
        base.extend(["sshpass", "-p", password])
    return base


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory)
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    destination = Path(args.output_dir or (ROOT / "outputs" / "deployment" / args.run_id)).resolve()
    local_addresses = _local_addresses()
    rows: list[dict[str, Any]] = []
    for node in inventory.nodes:
        artifact_root = str(summary["resolved_paths"].get(f"{node.name}_artifact_root") or "")
        if not artifact_root:
            raise RuntimeError(f"missing artifact root for {node.name}")
        source = f"{artifact_root.rstrip('/')}/{args.run_id}"
        target = destination / str(node.name)
        host = str(node.ssh_host or node.host)
        local_mode = str(node.host) in local_addresses or host in local_addresses
        if local_mode:
            command = ["copytree", source, str(target)]
        else:
            command = [
                *_scp_base(),
                "scp",
                "-r",
                "-P",
                str(node.port),
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                f"{node.ssh_user}@{host}:{source}/.",
                str(target),
            ]
        row = {"node_name": node.name, "source": source, "target": str(target), "mode": "local" if local_mode else "scp", "command": command}
        if not args.apply:
            row["status"] = "DRY_RUN"
            rows.append(row)
            continue
        target.mkdir(parents=True, exist_ok=True)
        if local_mode:
            if not Path(source).is_dir():
                row["status"] = "MISSING"
            else:
                shutil.copytree(source, target, dirs_exist_ok=True)
                row["status"] = "PASS"
        else:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            row["returncode"] = completed.returncode
            row["output"] = (completed.stdout + completed.stderr)[-2000:]
            row["status"] = "PASS" if completed.returncode == 0 else "FAIL"
        rows.append(row)
    status = "DRY_RUN" if not args.apply else ("PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL")
    payload = {
        "schema_version": "routersense.deploy.collect.v2",
        "apply_mode": bool(args.apply),
        "inventory": summary,
        "run_id": args.run_id,
        "output_dir": str(destination),
        "nodes": rows,
        "status": status,
    }
    if args.apply:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "collection_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if status in {"DRY_RUN", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
