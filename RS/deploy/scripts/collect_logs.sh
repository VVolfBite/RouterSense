#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
INVENTORY="${1:-$ROOT/deploy/inventory/hosts.local.yaml}"

python - "$INVENTORY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from routesense.topology import inventory_cli_summary, load_inventory

inventory = load_inventory(Path(sys.argv[1]))
print(json.dumps({"dry_run": True, "inventory": inventory_cli_summary(inventory)}, indent=2))
PY
