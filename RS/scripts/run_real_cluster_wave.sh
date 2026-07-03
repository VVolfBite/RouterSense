#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

INVENTORY="${1:-$DEFAULT_INVENTORY}"
STRATEGY="${2:-U_gated_maxweight_matching_atomic}"
EXECUTION_MODE="${3:-scheduled_transport}"
MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}"
PROMPT="${PROMPT:-Explain mixture-of-experts routing in one paragraph.}"
LAYER_INDEX="${LAYER_INDEX:-0}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-20}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
PRECISION="${PRECISION:-fp16}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${STRATEGY}-${EXECUTION_MODE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/rs_wave_runs/${RUN_ID}}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
REMOTE_RESULT_GLOB="${EXECUTION_MODE}_${STRATEGY}_layer*.json"

mkdir -p "$OUTPUT_ROOT"

readarray -t META < <(
  cd "$ROOT" && "$PYTHON_BIN" - "$INVENTORY" "$MODEL" <<'PY'
from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

from rs.topology import load_inventory, resolve_model_path_for_node, resolve_node_rs_root

inventory = load_inventory(Path(sys.argv[1]))
model_id = sys.argv[2]
local_ips = {"127.0.0.1", "localhost"}
try:
    local_ips.update(subprocess.check_output(["hostname", "-I"], text=True).split())
except Exception:
    pass
try:
    local_ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
except Exception:
    pass

local_node = None
remote_node = None
for node in inventory.nodes:
    ssh_host = node.ssh_host or node.host
    if node.host in local_ips or ssh_host in local_ips:
        local_node = node
    else:
        remote_node = node

if local_node is None or remote_node is None:
    raise RuntimeError("inventory must contain one local node and one remote node")

master = next(node for node in inventory.nodes if node.name == inventory.rendezvous.master_node)
fields = [
    ("LOCAL_NODE_NAME", local_node.name),
    ("LOCAL_NODE_RANK", str(local_node.node_rank)),
    ("LOCAL_NODE_HOST", local_node.host),
    ("LOCAL_RS_ROOT", str(resolve_node_rs_root(inventory, local_node.name))),
    ("LOCAL_MODEL_PATH", str(resolve_model_path_for_node(inventory, local_node.name, model_id) or "")),
    ("REMOTE_NODE_NAME", remote_node.name),
    ("REMOTE_NODE_RANK", str(remote_node.node_rank)),
    ("REMOTE_NODE_HOST", remote_node.host),
    ("REMOTE_SSH_HOST", str(remote_node.ssh_host or remote_node.host)),
    ("REMOTE_PORT", str(remote_node.port)),
    ("REMOTE_USER", remote_node.ssh_user),
    ("REMOTE_RS_ROOT", str(resolve_node_rs_root(inventory, remote_node.name))),
    ("REMOTE_MODEL_PATH", str(resolve_model_path_for_node(inventory, remote_node.name, model_id) or "")),
    ("NNODES", str(len(inventory.nodes))),
    ("MASTER_ADDR", master.host),
    ("MASTER_PORT", str(inventory.rendezvous.master_port)),
]
for key, value in fields:
    print(f"{key}={value}")
PY
)

for line in "${META[@]}"; do
  eval "export ${line}"
done

REMOTE_LOG="${REMOTE_LOG:-${REMOTE_OUTPUT_ROOT}/${REMOTE_NODE_NAME}.log}"
LOCAL_LOG="${LOCAL_LOG:-${OUTPUT_ROOT}/${LOCAL_NODE_NAME}.log}"

if [[ -z "${RSSH_PASSWORD:-${SSHPASS:-}}" ]]; then
  echo "missing SSH password; set RSSH_PASSWORD or SSHPASS" >&2
  exit 1
fi
SSH_PASSWORD="${RSSH_PASSWORD:-${SSHPASS:-}}"

if [[ -z "${LOCAL_RS_ROOT:-}" || -z "${REMOTE_RS_ROOT:-}" ]]; then
  echo "failed to resolve RS roots from inventory" >&2
  exit 1
fi

REMOTE_MODEL_ARG=()
if [[ -n "${REMOTE_MODEL_PATH:-}" ]]; then
  REMOTE_MODEL_ARG=(--model-path "$REMOTE_MODEL_PATH")
fi
LOCAL_MODEL_ARG=()
if [[ -n "${LOCAL_MODEL_PATH:-}" ]]; then
  LOCAL_MODEL_ARG=(--model-path "$LOCAL_MODEL_PATH")
fi

REMOTE_CMD="$(cat <<EOF
set -euo pipefail
mkdir -p '$REMOTE_OUTPUT_ROOT'
cd '$REMOTE_RS_ROOT'
PYTHONPATH=src NCCL_IB_DISABLE=1 TORCHDISTRIBUTED_DEBUG=DETAIL \
torchrun --nnodes=$NNODES --nproc_per_node=1 --node_rank=$REMOTE_NODE_RANK \
  --rdzv-backend=c10d --rdzv-id=rs-wave-$RUN_ID --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  experiments/distributed/exp_wave_execution.py \
  --inventory '$INVENTORY' \
  --node-name '$REMOTE_NODE_NAME' \
  --device-map '$DEVICE_MAP' \
  --max-memory-gb '$MAX_MEMORY_GB' \
  --precision '$PRECISION' \
  --model '$MODEL' \
  ${REMOTE_MODEL_ARG[*]:-} \
  --strategy '$STRATEGY' \
  --execution-mode '$EXECUTION_MODE' \
  --prompt '$PROMPT' \
  --layer-index '$LAYER_INDEX' \
  --output-dir '$REMOTE_OUTPUT_ROOT' \
  > '$REMOTE_LOG' 2>&1
EOF
)"

LOCAL_CMD=(
  torchrun
  --nnodes="$NNODES"
  --nproc_per_node=1
  --node_rank="$LOCAL_NODE_RANK"
  --rdzv-backend=c10d
  --rdzv-id="rs-wave-$RUN_ID"
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
  experiments/distributed/exp_wave_execution.py
  --inventory "$INVENTORY"
  --node-name "$LOCAL_NODE_NAME"
  --device-map "$DEVICE_MAP"
  --max-memory-gb "$MAX_MEMORY_GB"
  --precision "$PRECISION"
  --model "$MODEL"
  "${LOCAL_MODEL_ARG[@]}"
  --strategy "$STRATEGY"
  --execution-mode "$EXECUTION_MODE"
  --prompt "$PROMPT"
  --layer-index "$LAYER_INDEX"
  --output-dir "$OUTPUT_ROOT"
)

sshpass -p "$SSH_PASSWORD" ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -p "$REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_SSH_HOST" \
  "bash -lc $(printf '%q' "mkdir -p '$REMOTE_OUTPUT_ROOT' && rm -f '$REMOTE_LOG' && nohup bash -lc $(printf '%q' "$REMOTE_CMD") >/dev/null 2>&1 & echo \$!")" \
  > "$OUTPUT_ROOT/remote.pid"

sleep 5

(
  cd "$LOCAL_RS_ROOT"
  PYTHONPATH=src NCCL_IB_DISABLE=1 TORCHDISTRIBUTED_DEBUG=DETAIL "${LOCAL_CMD[@]}"
) | tee "$LOCAL_LOG"

REMOTE_PID="$(tr -d '[:space:]' < "$OUTPUT_ROOT/remote.pid")"
for _ in $(seq 1 60); do
  if sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_SSH_HOST" \
    "ls '$REMOTE_OUTPUT_ROOT'/$REMOTE_RESULT_GLOB >/dev/null 2>&1 || ! kill -0 '$REMOTE_PID' 2>/dev/null"; then
    break
  fi
  sleep 2
done

sshpass -p "$SSH_PASSWORD" scp \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -P "$REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_SSH_HOST:$REMOTE_LOG" \
  "$OUTPUT_ROOT/${REMOTE_NODE_NAME}.log" >/dev/null

REMOTE_RESULT_PATH="$(
  sshpass -p "$SSH_PASSWORD" ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p "$REMOTE_PORT" \
    "$REMOTE_USER@$REMOTE_SSH_HOST" \
    "ls -t '$REMOTE_OUTPUT_ROOT'/$REMOTE_RESULT_GLOB | head -n 1"
)"
if [[ -z "$REMOTE_RESULT_PATH" ]]; then
  echo "remote result json not found under $REMOTE_OUTPUT_ROOT" >&2
  exit 1
fi

sshpass -p "$SSH_PASSWORD" scp \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -P "$REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_SSH_HOST:$REMOTE_RESULT_PATH" \
  "$OUTPUT_ROOT/result.json" >/dev/null

"$PYTHON_BIN" - "$OUTPUT_ROOT/result.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ranks = payload["ranks"]
summary = {
    "run": payload["run"],
    "rank_count": len(ranks),
    "transport": ranks[0]["execution"]["wave_execution"]["transport"],
    "dispatch_comm_ms": [round(rank["execution"]["wave_execution"]["dispatch_comm_ms"], 3) for rank in ranks],
    "combine_comm_ms": [round(rank["execution"]["wave_execution"]["combine_comm_ms"], 3) for rank in ranks],
    "max_abs_error": max(float(rank["execution"]["correctness"]["max_abs_error"]) for rank in ranks),
    "token_conservation_pass": all(bool(rank["execution"]["correctness"]["token_conservation_pass"]) for rank in ranks),
    "gate_weight_conservation_pass": all(bool(rank["execution"]["correctness"]["gate_weight_conservation_pass"]) for rank in ranks),
}
print(json.dumps(summary, indent=2))
PY
