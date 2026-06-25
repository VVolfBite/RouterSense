#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bootstrap_and_inspect.log"

MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0125}"
HF_HOME="${HF_HOME:-$ROOT_DIR/hf_cache}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_CMD="${PYTHON_CMD:-python3}"

mkdir -p "$HF_HOME"
mkdir -p "$ROOT_DIR/outputs"

exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ ! -f "$ROOT_DIR/scripts/setup_env.sh" ]]; then
  log "scripts/setup_env.sh not found" >&2
  exit 1
fi

log "Project root: $ROOT_DIR"
log "Model ID: $MODEL_ID"
log "HF_HOME: $HF_HOME"
log "Python binary: $PYTHON_BIN"

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  log "Creating virtual environment"
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
fi

source "$ROOT_DIR/.venv/bin/activate"

bash "$ROOT_DIR/scripts/setup_env.sh"

python - <<'PY'
from routesense_poc1.output_archive import archive_previous_outputs
archive_previous_outputs("outputs")
print("archived previous outputs")
PY

export HF_HOME
export MODEL_ID

log "Running GPU preflight"
"$PYTHON_CMD" "$ROOT_DIR/scripts/preflight_gpu.py"

if ! "$PYTHON_CMD" - <<'PY'
import importlib.util
print(importlib.util.find_spec('torch'))
PY
then
  log "PyTorch import failed"
  exit 1
fi

"$PYTHON_CMD" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("CUDA available")
PY

log "Running model inspection"
"$PYTHON_CMD" "$ROOT_DIR/scripts/inspect_model.py"

if [[ ! -f "$ROOT_DIR/outputs/model_inspection.json" ]]; then
  echo "model_inspection.json was not created" >&2
  exit 1
fi

"$PYTHON_CMD" - <<'PY'
import json
from pathlib import Path
inspection = json.loads(Path("outputs/model_inspection.json").read_text(encoding="utf-8"))
if inspection.get("status") != "ok":
    raise SystemExit("inspection did not complete successfully")
print("inspection status ok")
PY

log "环境与模型检查完成。"
log "请查看："
log "- outputs/preflight_gpu.json"
log "- outputs/environment.json"
log "- outputs/model_inspection.json"
log "下一步：根据真实 MoE layer.forward 结构实现 OLMoE adapter。"
log "在 adapter 未完成前，不要运行真实 trace 或 ablation。"
