#!/usr/bin/env python3
from __future__ import annotations

"""Ensure every inventory node has the same verified model snapshot."""

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

from rs.topology import DEFAULT_DEPLOYMENT_MODEL_ID, inventory_cli_summary, load_inventory
from scripts.deploy.launch_remote import _local_addresses, _run_node


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_DEPLOYMENT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory)
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    rows: list[dict[str, Any]] = []
    for node in inventory.nodes:
        remote_root = str(summary["resolved_paths"].get(f"{node.name}_remote_rs_root") or "")
        if not remote_root:
            raise RuntimeError(f"missing remote root for {node.name}")
        remote_inventory = f"{remote_root}/deploy/inventory/{inventory_path.name}"
        parts = [
            "bash",
            "scripts/deploy/prefetch_model.sh",
            remote_inventory,
            str(node.name),
            "--model-id",
            str(args.model_id),
        ]
        if args.apply:
            parts.append("--apply")
        if args.revision:
            parts.extend(["--revision", str(args.revision)])
        if args.local_files_only:
            parts.append("--local-files-only")
        command = f"set -euo pipefail; cd {shlex.quote(remote_root)}; " + " ".join(shlex.quote(item) for item in parts)
        rows.append({"node_name": node.name, "command": command, "status": "DRY_RUN"})
    payload: dict[str, Any] = {
        "schema_version": "routersense.deploy.model_sync.v2",
        "apply_mode": bool(args.apply),
        "inventory": summary,
        "model_id": str(args.model_id),
        "revision": args.revision or "default",
        "nodes": rows,
    }
    if not args.apply:
        payload["status"] = "DRY_RUN"
        payload["MODEL_CACHE_PARITY_PASS"] = False
        print(json.dumps(payload, indent=2))
        return 0

    local_addresses = _local_addresses()
    results = []
    node_by_name = {str(node.name): node for node in inventory.nodes}
    for row in rows:
        output = _run_node(node_by_name[str(row["node_name"])], str(row["command"]), local_addresses=local_addresses)
        result = json.loads(output)
        after = dict(result.get("after") or result.get("before") or {})
        results.append(
            {
                **row,
                "status": str(result.get("status", "FAIL")),
                "model_path": after.get("model_path"),
                "required_files_present": bool(after.get("required_files_present", False)),
                "manifest_hash": after.get("manifest_hash"),
                "total_size_bytes": after.get("total_size_bytes"),
            }
        )
    hashes = {str(row["manifest_hash"]) for row in results if row["required_files_present"]}
    parity = bool(results and all(row["required_files_present"] for row in results) and len(hashes) == 1)
    payload["nodes"] = results
    payload["MODEL_CACHE_PARITY_PASS"] = parity
    payload["status"] = "PASS" if parity else "FAIL"
    print(json.dumps(payload, indent=2))
    return 0 if parity else 2


if __name__ == "__main__":
    raise SystemExit(main())
