#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID_DIR="${PID_DIR:-$ROOT/deploy/logs}"

shopt -s nullglob
for pidfile in "$PID_DIR"/*.pid; do
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "stopped pid=$pid from $pidfile"
  fi
done

