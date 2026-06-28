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

from routesense.topology import load_inventory, render_torchrun_dry_run

inventory = load_inventory(Path(sys.argv[1]))
print(json.dumps({"dry_run": True, **render_torchrun_dry_run(inventory)}, indent=2))
PY

