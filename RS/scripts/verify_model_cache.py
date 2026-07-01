#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.topology import load_inventory, resolve_node_model_cache


def main(argv: list[str] | None = None) -> int:
    inventory_path = Path(argv[0] if argv else sys.argv[1])
    node_name = argv[1] if argv and len(argv) > 1 else (sys.argv[2] if len(sys.argv) > 2 else "node0")
    inventory = load_inventory(inventory_path)
    cache = resolve_node_model_cache(inventory, node_name)
    payload = {
        "node_name": node_name,
        "cache_path": str(cache or ""),
        "required_files_present": bool(cache),
        "tokenizer_ready": bool(cache),
        "config_ready": bool(cache),
        "weights_ready": bool(cache),
        "gpu_runtime_attempted": False,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
