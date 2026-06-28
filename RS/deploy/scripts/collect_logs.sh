#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"
echo "[dry-run] would collect deployment logs using inventory: $INVENTORY"

