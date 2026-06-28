#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.topology import load_inventory, resolve_node_model_cache


def main(argv: list[str] | None = None) -> int:
    inventory_path = Path(argv[0] if argv else sys.argv[1])
    inventory = load_inventory(inventory_path)
    payload = {"nodes": []}
    for node in inventory.nodes:
        cache = resolve_node_model_cache(inventory, node.name)
        manifest = []
        if cache is not None and cache.exists():
            for path in sorted(cache.rglob("*")):
                if path.is_file():
                    manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(cache)}")
        payload["nodes"].append(
            {
                "node_name": node.name,
                "node_rank": node.node_rank,
                "model_cache": str(cache or ""),
                "model_id": "allenai/OLMoE-1B-7B-0924",
                "required_files_present": bool(cache and cache.exists()),
                "manifest_hash": hashlib.sha256("\n".join(manifest).encode("utf-8")).hexdigest() if manifest else "missing",
            }
        )
    payload["MODEL_CACHE_MISSING"] = any(not item["required_files_present"] for item in payload["nodes"])
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
