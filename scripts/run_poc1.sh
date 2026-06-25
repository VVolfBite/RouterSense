#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "Virtual environment not found. Run scripts/setup_env.sh first." >&2
  exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"

python3 "$ROOT_DIR/scripts/preflight_gpu.py"
set +e
python3 "$ROOT_DIR/scripts/check_adapter_ready.py"
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "真实 OLMoE adapter 尚未完成。"
  echo "环境和模型检查已可运行，但当前不能生成可信的 Router trace 或 ΔNLL。"
  echo "请先根据 outputs/selected_moe_layer_forward.txt 实现 adapter。"
  exit 3
fi

python3 "$ROOT_DIR/scripts/run_action.py" trace --text "The history of science is a story of" --layer auto
python3 "$ROOT_DIR/scripts/run_action.py" ablate --text "The history of science is a story of" --layer auto --rank 0
