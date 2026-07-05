#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
INVENTORY="${1:-$DEFAULT_INVENTORY}"

"$PYTHON_BIN" - "$INVENTORY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from rs.topology import inventory_cli_summary, load_inventory

inventory_path = Path(sys.argv[1])
inventory = load_inventory(inventory_path)
print(json.dumps({"dry_run": True, "inventory": inventory_cli_summary(inventory, inventory_path=inventory_path)}, indent=2))
PY
