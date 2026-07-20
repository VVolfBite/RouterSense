#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-}"
FORCE="${2:-}"
case "$MODE" in
  1x4) TEMPLATE="$ROOT/deploy/inventory/hosts.1x4.example.yaml" ;;
  2x2) TEMPLATE="$ROOT/deploy/inventory/hosts.2x2.example.yaml" ;;
  *)
    echo "Usage: bash scripts/deploy/init_inventory.sh {1x4|2x2} [--force]" >&2
    exit 2
    ;;
esac
TARGET="$ROOT/deploy/inventory/hosts.local.yaml"
if [[ -e "$TARGET" && "$FORCE" != "--force" ]]; then
  echo "Refusing to overwrite $TARGET. Use --force only when replacement is intended." >&2
  exit 2
fi
cp "$TEMPLATE" "$TARGET"
echo "Created $TARGET from $(basename "$TEMPLATE")."
echo "Edit only host, optional ssh_host, gpu_count, and model_path."
