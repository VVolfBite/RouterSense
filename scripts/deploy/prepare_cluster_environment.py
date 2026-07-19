#!/usr/bin/env python3
from __future__ import annotations

"""Prepare RouterSense Python dependencies on all inventory nodes."""

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.topology import inventory_cli_summary, load_inventory
from scripts.deploy.launch_remote import _local_addresses, _run_node


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-dev", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory).resolve()
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    local_addresses = _local_addresses()
    rows: list[dict[str, Any]] = []
    for node in inventory.nodes:
        remote_root = str(summary["resolved_paths"].get(f"{node.name}_remote_rs_root") or "")
        if not remote_root:
            raise RuntimeError(f"missing remote root for {node.name}")
        remote_inventory = f"{remote_root}/deploy/inventory/{inventory_path.name}"
        parts = [
            "bash",
            "scripts/deploy/prepare_node.sh",
            remote_inventory,
            str(node.name),
        ]
        if args.apply:
            parts.append("--apply")
        if args.include_dev:
            parts.append("--include-dev")
        command = f"set -euo pipefail; cd {shlex.quote(remote_root)}; " + " ".join(shlex.quote(item) for item in parts)
        row: dict[str, Any] = {"node_name": node.name, "command": command}
        if not args.apply:
            row["status"] = "DRY_RUN"
        else:
            try:
                output = _run_node(node, command, local_addresses=local_addresses)
                payload = json.loads(output)
                row["payload"] = payload
                row["status"] = "PASS" if payload.get("source_ready") else "FAIL"
            except Exception as exc:
                row["status"] = "FAIL"
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    status = "DRY_RUN" if not args.apply else ("PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL")
    payload = {
        "schema_version": "routersense.deploy.environment.v2",
        "inventory": summary,
        "apply_mode": bool(args.apply),
        "include_dev": bool(args.include_dev),
        "targets": rows,
        "status": status,
    }
    print(json.dumps(payload, indent=2))
    return 0 if status in {"DRY_RUN", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
