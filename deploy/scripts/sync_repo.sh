#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"
APPLY=false
for arg in "$@"; do
  [[ "$arg" == "--apply" ]] && APPLY=true
done

if [[ "$APPLY" != true ]]; then
  echo "[dry-run] would sync repo to controller and executor from: $INVENTORY"
  echo "[dry-run] excluded: .env, inventory local files, model cache, artifacts, outputs, logs"
  exit 0
fi

echo "apply mode not executed in this repository snapshot"
exit 1

