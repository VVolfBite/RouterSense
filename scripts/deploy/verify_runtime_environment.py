#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed GPU framework/runtime preflight on every inventory node."""

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
    parser.add_argument("--python-bin", default="python3")
    return parser.parse_args(argv)


def _remote_probe(*, remote_root: str, remote_inventory: str, node_name: str, model_id: str, python_bin: str, target_gpu_count: int) -> str:
    resolver = (
        f"{shlex.quote(python_bin)} scripts/verify/verify_model_cache.py "
        f"{shlex.quote(remote_inventory)} {shlex.quote(node_name)} "
        f"--model-id {shlex.quote(model_id)} --print-model-path"
    )
    code = r'''
import importlib.util, json, os, subprocess, sys
required = ["numpy", "numba", "scipy", "sklearn", "yaml", "transformers", "accelerate", "safetensors", "sentencepiece"]
modules = {name: bool(importlib.util.find_spec(name)) for name in required}
command = [sys.executable, "experiments/online/support/environment_validation.py", "--model", os.environ["RS_MODEL_PATH"]]
completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
try:
    runtime = json.loads(completed.stdout)
except Exception:
    runtime = {"status": "blocked_environment", "reason": "invalid_environment_validation_output", "output_tail": completed.stdout[-4000:]}
visible = int(runtime.get("visible_gpu_count", 0) or 0)
missing_python = sorted(name for name, available in modules.items() if not available)
status = "PASS" if completed.returncode == 0 and runtime.get("status") == "ready" and visible >= TARGET and not missing_python else "FAIL"
print(json.dumps({
    "status": status,
    "target_gpu_count": TARGET,
    "visible_gpu_count": visible,
    "missing_python_modules": missing_python,
    "python_modules": modules,
    "runtime_validation": runtime,
}, indent=2))
'''.replace("TARGET", str(int(target_gpu_count)))
    return (
        f"set -euo pipefail; cd {shlex.quote(remote_root)}; "
        f"RS_MODEL_PATH=$({resolver}); export RS_MODEL_PATH; "
        f"{shlex.quote(python_bin)} - <<'PY'\n{code}\nPY"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory_path = Path(args.inventory)
    inventory = load_inventory(inventory_path)
    summary = inventory_cli_summary(inventory, inventory_path=inventory_path)
    local_addresses = _local_addresses()
    rows: list[dict[str, Any]] = []
    for node in inventory.nodes:
        remote_root = str(summary["resolved_paths"].get(f"{node.name}_remote_rs_root") or "")
        remote_inventory = f"{remote_root}/deploy/inventory/{inventory_path.name}"
        command = _remote_probe(
            remote_root=remote_root,
            remote_inventory=remote_inventory,
            node_name=str(node.name),
            model_id=str(args.model_id),
            python_bin=str(args.python_bin),
            target_gpu_count=int(node.target_gpu_count),
        )
        row: dict[str, Any] = {"node_name": str(node.name), "command": command, "status": "DRY_RUN"}
        if args.apply:
            try:
                payload = json.loads(_run_node(node, command, local_addresses=local_addresses))
                row.update(payload)
                row["status"] = "PASS" if payload.get("status") == "PASS" else "FAIL"
            except Exception as exc:
                row.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    status = "DRY_RUN" if not args.apply else ("PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL")
    payload = {
        "schema_version": "routersense.deploy.runtime_environment_preflight.v1",
        "apply_mode": bool(args.apply),
        "model_id": str(args.model_id),
        "inventory": summary,
        "nodes": rows,
        "status": status,
    }
    print(json.dumps(payload, indent=2))
    return 0 if status in {"DRY_RUN", "PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
