#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
INVENTORY="${1:-$DEFAULT_INVENTORY}"
NODE_NAME="${2:-node0}"
APPLY=false
INCLUDE_DEV=false
PIP_INDEX_URL_VALUE="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

for arg in "$@"; do
  [[ "$arg" == "--apply" ]] && APPLY=true
  [[ "$arg" == "--include-dev" ]] && INCLUDE_DEV=true
done

"$PYTHON_BIN" - "$INVENTORY" "$NODE_NAME" "$APPLY" "$INCLUDE_DEV" "$PIP_INDEX_URL_VALUE" "$ROOT" "$PYTHON_BIN" <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rs.topology import DEFAULT_DEPLOYMENT_MODEL_ID, inspect_model_cache, inventory_cli_summary, load_inventory, resolve_node_artifact_root, resolve_node_model_cache, resolve_node_rs_root

inventory_path = Path(sys.argv[1])
inventory = load_inventory(inventory_path)
node_name = sys.argv[2]
apply_mode = sys.argv[3].lower() == "true"
include_dev = sys.argv[4].lower() == "true"
pip_index_url = sys.argv[5]
root = Path(sys.argv[6])
python_bin = sys.argv[7]
remote_rs_root = Path(resolve_node_rs_root(inventory, node_name) or root)
model_cache = Path(resolve_node_model_cache(inventory, node_name) or "")
artifact_root = Path(resolve_node_artifact_root(inventory, node_name) or "")

deps = ["transformers", "accelerate", "safetensors", "sentencepiece", "huggingface_hub"]
if include_dev:
    deps.extend(["pytest", "pyyaml"])

source_ready = bool(
    remote_rs_root.is_dir()
    and (remote_rs_root / "pyproject.toml").is_file()
    and (remote_rs_root / "src" / "rs" / "__init__.py").is_file()
)
model_inspection = inspect_model_cache(model_cache, DEFAULT_DEPLOYMENT_MODEL_ID)
payload = {
    "node_name": node_name,
    "inventory": inventory_cli_summary(inventory, inventory_path=inventory_path),
    "remote_rs_root": str(remote_rs_root),
    "model_cache": str(model_cache),
    "artifact_root": str(artifact_root),
    "apply_mode": apply_mode,
    "include_dev": include_dev,
    "pip_index_url": pip_index_url,
    "install_commands": [
        f"PIP_INDEX_URL={pip_index_url} {python_bin} -m pip install {' '.join(deps)}",
        f"{python_bin} -m pip install --no-deps -e {remote_rs_root}",
    ],
    "source_ready": source_ready,
    **model_inspection.to_dict(),
    "deployment_prerequisites_ready": bool(source_ready and model_inspection.required_files_present),
    "gpu_runtime_attempted": False,
}

if apply_mode:
    env = dict(os.environ)
    env["PIP_INDEX_URL"] = pip_index_url
    payload["applied_steps"] = []

    def run_step(step: str, command: list[str], details: dict[str, object]) -> bool:
        completed = subprocess.run(
            command,
            check=False,
            cwd=remote_rs_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        row = {
            "step": step,
            **details,
            "returncode": int(completed.returncode),
            "output_tail": (completed.stdout or "")[-4000:],
            "status": "PASS" if completed.returncode == 0 else "FAIL",
        }
        payload["applied_steps"].append(row)
        return completed.returncode == 0

    ok = run_step(
        "pip_install_deps",
        [python_bin, "-m", "pip", "install", *deps],
        {"packages": deps},
    )
    if ok:
        ok = run_step(
            "pip_install_editable",
            [python_bin, "-m", "pip", "install", "--no-deps", "-e", str(remote_rs_root)],
            {"path": str(remote_rs_root)},
        )

    if ok:
        verify_code = (
            "import importlib;"
            "mods=['torch','transformers','accelerate','safetensors','sentencepiece','yaml','rs'];"
            "out={m:getattr(importlib.import_module(m),'__version__','n/a') for m in mods};"
            "print(out)"
        )
        verify = subprocess.run(
            [python_bin, "-c", verify_code],
            check=False,
            cwd=remote_rs_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        payload["verification"] = {
            "returncode": int(verify.returncode),
            "output": (verify.stdout or "").strip(),
            "status": "PASS" if verify.returncode == 0 else "FAIL",
        }
        ok = verify.returncode == 0

    payload["status"] = "PASS" if ok else "FAIL"
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if ok else 2)

payload["status"] = "DRY_RUN"
print(json.dumps(payload, indent=2))
PY
