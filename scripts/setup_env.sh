#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
fi

source "$ROOT_DIR/.venv/bin/activate"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/legacy/shared/requirements-gpu.txt"

"$PYTHON_BIN" - <<'PY'
import importlib
for name in ["torch", "transformers", "accelerate", "numpy", "yaml"]:
    mod = importlib.import_module(name)
    print(f"{name}: OK")
PY

"$PYTHON_BIN" - <<'PY'
import importlib.util
import torch
print("Python", torch.__version__)
print("CUDA availability", torch.cuda.is_available())
PY
