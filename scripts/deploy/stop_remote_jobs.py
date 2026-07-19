#!/usr/bin/env python3
from __future__ import annotations

"""Stop a detached RouterSense deployment run on all inventory nodes."""

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
    parser.add_argument("--run-id", required=True)
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
        artifact_root = str(summary["resolved_paths"].get(f"{node.name}_artifact_root") or "")
        pidfile = f"{artifact_root.rstrip('/')}/{args.run_id}/logs/{node.name}.pid"
        command = (
            f"set -euo pipefail; if [ ! -f {shlex.quote(pidfile)} ]; then echo MISSING; exit 0; fi; "
            f"pid=$(cat {shlex.quote(pidfile)}); "
            "if ! printf '%s' \"$pid\" | grep -Eq '^[0-9]+$'; then echo INVALID; exit 2; fi; "
            "if kill -0 \"$pid\" 2>/dev/null; then kill -TERM -- \"-$pid\" 2>/dev/null || kill -TERM \"$pid\"; echo STOPPED; else echo NOT_RUNNING; fi"
        )
        row = {"node_name": node.name, "pidfile": pidfile, "command": command}
        if not args.apply:
            row["status"] = "DRY_RUN"
        else:
            result = _run_node(node, command, local_addresses=local_addresses).strip()
            row["result"] = result
            row["status"] = "PASS" if result in {"STOPPED", "NOT_RUNNING", "MISSING"} else "FAIL"
        rows.append(row)
    status = "DRY_RUN" if not args.apply else ("PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL")
    payload = {"schema_version": "routersense.deploy.stop.v2", "apply_mode": bool(args.apply), "run_id": args.run_id, "nodes": rows, "status": status}
    print(json.dumps(payload, indent=2))
    return 0 if status in {"DRY_RUN", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
