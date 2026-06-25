#!/usr/bin/env bash
# Linux GPU server entry point: create env, run preflight, inspect status, and real trace.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-allenai/OLMoE-1B-7B-0125}"
TEXT_INPUT="${TEXT_INPUT:-The history of science is a story of}"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/run_poc1_linux.log"
OUTPUT_DIR="$ROOT_DIR/outputs/poc1_trace_single"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  "$@"
  log "DONE $name"
}

if [[ ! -f "$ROOT_DIR/scripts/setup_env.sh" ]]; then
  echo "scripts/setup_env.sh not found" >&2
  exit 1
fi

run_step "setup_env" bash "$ROOT_DIR/scripts/setup_env.sh"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
run_step "preflight" "$PYTHON_BIN" "$ROOT_DIR/scripts/preflight_gpu.py"
run_step "inspect" "$PYTHON_BIN" "$ROOT_DIR/scripts/inspect_model.py"
run_step "trace" "$PYTHON_BIN" "$ROOT_DIR/scripts/run_action.py" trace --text "$TEXT_INPUT" --layer auto --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"
log "SKIP ablate: real route ablation is not implemented in this project state."

log "Model ID: $MODEL_ID"
log "Text input: $TEXT_INPUT"
log "Results directory: $OUTPUT_DIR"
log "Log file: $LOG_FILE"
log "Generated files:"
find "$OUTPUT_DIR" -maxdepth 1 -type f | sort | sed "s#^#  #"

echo "Linux PoC1 deployment/run entry completed."
