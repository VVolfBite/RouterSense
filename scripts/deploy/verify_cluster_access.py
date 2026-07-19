#!/usr/bin/env python3
from __future__ import annotations

"""Verify SSH/key access and basic Python availability for inventory nodes."""

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory)
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    rows: list[dict[str, Any]] = []
    local_addresses = _local_addresses()
    for node in inventory.nodes:
        remote_root = str(summary["resolved_paths"].get(f"{node.name}_remote_rs_root") or "")
        command = (
            "set -euo pipefail; "
            "printf '__HOST__%s\\n' \"$(hostname)\"; "
            "printf '__PYTHON__%s\\n' \"$(python3 -V 2>&1)\"; "
            f"if [ -d {shlex.quote(remote_root + '/.git')} ]; then "
            f"printf '__REPO__%s\\n' \"$(git -C {shlex.quote(remote_root)} rev-parse HEAD)\"; "
            "else printf '__REPO__MISSING\\n'; fi"
        )
        row = {"node_name": node.name, "host": node.ssh_host or node.host, "port": node.port, "command": command}
        if not args.apply:
            row["status"] = "DRY_RUN"
        else:
            try:
                output = _run_node(node, command, local_addresses=local_addresses)
                row["output"] = output
                row["status"] = "PASS" if "__HOST__" in output and "__PYTHON__" in output else "FAIL"
            except Exception as exc:
                row["status"] = "FAIL"
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    status = "DRY_RUN" if not args.apply else ("PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL")
    payload = {
        "schema_version": "routersense.deploy.access.v2",
        "apply_mode": bool(args.apply),
        "inventory": summary,
        "nodes": rows,
        "tcp_probe": {"status": "deferred_until_torchrun_rendezvous"},
        "status": status,
    }
    print(json.dumps(payload, indent=2))
    return 0 if status in {"DRY_RUN", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
