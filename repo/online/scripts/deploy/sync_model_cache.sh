#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
INVENTORY="${1:-$DEFAULT_INVENTORY}"
APPLY=false
for arg in "$@"; do
  [[ "$arg" == "--apply" ]] && APPLY=true
done

if [[ "$APPLY" != true ]]; then
  echo "[dry-run] would verify model cache parity for: $INVENTORY"
  echo "[dry-run] no transfer will be performed"
  exit 0
fi

echo "apply mode not executed in this repository snapshot"
exit 1
