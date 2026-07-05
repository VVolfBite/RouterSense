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

from rs.topology import inventory_cli_summary, load_inventory, resolve_node_artifact_root, resolve_node_model_cache, resolve_node_rs_root

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

deps = ["transformers", "accelerate", "safetensors", "sentencepiece"]
if include_dev:
    deps.extend(["pytest", "pyyaml"])

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
    "required_files_present": remote_rs_root.exists(),
    "tokenizer_ready": model_cache.exists(),
    "config_ready": model_cache.exists(),
    "weights_ready": model_cache.exists(),
    "gpu_runtime_attempted": False,
}

if apply_mode:
    env = dict(os.environ)
    env["PIP_INDEX_URL"] = pip_index_url
    payload["applied_steps"] = []

    subprocess.run(
        [python_bin, "-m", "pip", "install", *deps],
        check=True,
        cwd=remote_rs_root,
        env=env,
    )
    payload["applied_steps"].append({"step": "pip_install_deps", "packages": deps})

    subprocess.run(
        [python_bin, "-m", "pip", "install", "--no-deps", "-e", str(remote_rs_root)],
        check=True,
        cwd=remote_rs_root,
        env=env,
    )
    payload["applied_steps"].append({"step": "pip_install_editable", "path": str(remote_rs_root)})

    verify_code = (
        "import importlib;"
        "mods=['torch','transformers','accelerate','safetensors','sentencepiece','yaml','rs'];"
        "out={m:getattr(importlib.import_module(m),'__version__','n/a') for m in mods};"
        "print(out)"
    )
    verify_output = subprocess.check_output([python_bin, "-c", verify_code], text=True, cwd=remote_rs_root, env=env).strip()
    payload["verification"] = {"imports": verify_output}

print(json.dumps(payload, indent=2))
PY
