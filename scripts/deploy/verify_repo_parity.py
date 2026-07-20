#!/usr/bin/env python3
from __future__ import annotations

"""Verify clean commit and tracked-source parity on every inventory node."""

import argparse
import hashlib
import json
import shlex
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
from scripts.deploy.launch_remote import _local_addresses, _run_node

TRACKED_SCOPES = (
    "src",
    "experiments",
    "configs",
    "tests",
    "scripts",
    "deploy",
    "docs",
    "README.md",
    "pyproject.toml",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def _tree_hash(repo_root: Path) -> str:
    raw = subprocess.check_output(["git", "ls-files", "-z", *TRACKED_SCOPES], cwd=repo_root)
    files = sorted(item for item in raw.split(b"\0") if item)
    digest = hashlib.sha256()
    for relative_raw in files:
        relative = relative_raw.decode("utf-8")
        digest.update(relative_raw)
        path = repo_root / relative
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            # Dry-run parity must remain inspectable while a deployment cleanup
            # commit is being prepared. Apply mode still rejects a dirty tree.
            digest.update(b"\0MISSING\0")
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory).resolve()
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip())
    local_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    local_status = subprocess.check_output(["git", "status", "--short"], cwd=repo_root, text=True).strip()
    local_tree_hash = _tree_hash(repo_root)
    payload: dict[str, Any] = {
        "schema_version": "routersense.deploy.repo_parity.v2",
        "inventory": summary,
        "apply_mode": bool(args.apply),
        "local_head": local_head,
        "local_status": local_status,
        "local_tree_hash": local_tree_hash,
        "nodes": [],
    }
    if not args.apply:
        for node in inventory.nodes:
            payload["nodes"].append(
                {
                    "name": node.name,
                    "node_rank": node.node_rank,
                    "remote_root": summary["resolved_paths"].get(f"{node.name}_remote_rs_root"),
                    "status": "DRY_RUN",
                    "check": "remote git head, clean status, and canonical tracked-source tree hash",
                }
            )
        payload["verification_status"] = "DRY_RUN"
        payload["REPO_PARITY_PASS"] = False
        print(json.dumps(payload, indent=2))
        return 0

    local_addresses = _local_addresses()
    scope_args = " ".join(shlex.quote(item) for item in TRACKED_SCOPES)
    for node in inventory.nodes:
        remote_root = str(summary["resolved_paths"].get(f"{node.name}_remote_rs_root") or "")
        if not remote_root:
            raise RuntimeError(f"missing remote root for {node.name}")
        command = (
            f"set -euo pipefail; export LC_ALL=C; cd {shlex.quote(remote_root)}; "
            "printf '__HEAD__%s\\n' \"$(git rev-parse HEAD)\"; "
            "printf '__STATUS_BEGIN__\\n'; git status --short; printf '__STATUS_END__\\n'; "
            f"printf '__HASH__%s\\n' \"$(git ls-files -z {scope_args} | sort -z | "
            "while IFS= read -r -d '' f; do printf '%s' \"$f\"; cat \"$f\"; done | sha256sum | awk '{print $1}')\""
        )
        try:
            output = _run_node(node, command, local_addresses=local_addresses)
            head = output.split("__HEAD__", 1)[1].splitlines()[0].strip()
            status = output.split("__STATUS_BEGIN__", 1)[1].split("__STATUS_END__", 1)[0].strip()
            digest = output.split("__HASH__", 1)[1].splitlines()[0].strip()
            matches = head == local_head and not status and digest == local_tree_hash
            row = {
                "name": node.name,
                "node_rank": node.node_rank,
                "remote_root": remote_root,
                "head": head,
                "git_status": status,
                "tree_hash": digest,
                "matches_local": matches,
                "status": "PASS" if matches else "FAIL",
            }
        except Exception as exc:
            row = {
                "name": node.name,
                "node_rank": node.node_rank,
                "remote_root": remote_root,
                "matches_local": False,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
        payload["nodes"].append(row)
    parity = not local_status and bool(payload["nodes"]) and all(row["matches_local"] for row in payload["nodes"])
    payload["REPO_PARITY_PASS"] = parity
    payload["verification_status"] = "PASS" if parity else "FAIL"
    print(json.dumps(payload, indent=2))
    return 0 if parity else 2


if __name__ == "__main__":
    raise SystemExit(main())
