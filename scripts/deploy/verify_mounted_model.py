#!/usr/bin/env python3
from __future__ import annotations

"""Verify mounted model snapshots are structurally complete and locally loadable."""

import argparse
import json
import os
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
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _remote_probe(*, remote_root: str, remote_inventory: str, node_name: str, model_id: str, trust_remote_code: bool) -> str:
    resolver = (
        f"python3 scripts/verify/verify_model_cache.py {shlex.quote(remote_inventory)} {shlex.quote(node_name)} "
        f"--model-id {shlex.quote(model_id)} --print-model-path"
    )
    code = r'''
import json, os
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
p = Path(os.environ["RS_MODEL_PATH"]).resolve()
config_path = p / "config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
index_files = [p / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json") if (p / name).is_file()]
referenced = []
invalid_indexes = []
for index_path in index_files:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    values = sorted(set((payload.get("weight_map") or {}).values()))
    if not values:
        invalid_indexes.append(index_path.name)
    referenced.extend(values)
missing = [name for name in referenced if not (p / name).is_file() or (p / name).stat().st_size <= 0]
weight_files = sorted(
    file_path
    for file_path in list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    if file_path.is_file() and file_path.stat().st_size > 0
)
weights_ready = bool(weight_files) and not missing and not invalid_indexes
readable = []
for file_path in [config_path, *index_files, *weight_files[:2]]:
    with file_path.open("rb") as handle:
        readable.append({"name": file_path.name, "prefix_bytes": len(handle.read(4096))})
model_config = AutoConfig.from_pretrained(str(p), local_files_only=True, trust_remote_code=TRUST)
tokenizer = AutoTokenizer.from_pretrained(str(p), local_files_only=True, trust_remote_code=TRUST)
stat = os.statvfs(p)
print(json.dumps({
    "status": "PASS" if weights_ready else "FAIL",
    "model_path": str(p),
    "model_type": str(getattr(model_config, "model_type", "")),
    "tokenizer_class": type(tokenizer).__name__,
    "index_referenced_shards": len(referenced),
    "invalid_weight_indexes": invalid_indexes,
    "missing_index_shards": missing,
    "weight_file_count": len(weight_files),
    "readable_files": readable,
    "filesystem_free_bytes": int(stat.f_bavail * stat.f_frsize),
}, indent=2))
'''.replace("TRUST", "True" if trust_remote_code else "False")
    return (
        f"set -euo pipefail; cd {shlex.quote(remote_root)}; "
        f"RS_MODEL_PATH=$({resolver}); export RS_MODEL_PATH; "
        f"python3 - <<'PY'\n{code}\nPY"
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
            trust_remote_code=bool(args.trust_remote_code),
        )
        row: dict[str, Any] = {"node_name": str(node.name), "command": command, "status": "DRY_RUN"}
        if args.apply:
            try:
                output = _run_node(node, command, local_addresses=local_addresses)
                payload = json.loads(output)
                row.update(payload)
                row["status"] = "PASS" if payload.get("status") == "PASS" else "FAIL"
            except Exception as exc:
                row.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    status = "DRY_RUN" if not args.apply else ("PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL")
    payload = {
        "schema_version": "routersense.deploy.model_mount_preflight.v1",
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
