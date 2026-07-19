#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.topology import (
    DEFAULT_DEPLOYMENT_MODEL_ID,
    inspect_model_cache,
    load_inventory,
    resolve_node_model_cache,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local model snapshots and manifest parity for an inventory.")
    parser.add_argument("inventory")
    parser.add_argument("--model-id", default=DEFAULT_DEPLOYMENT_MODEL_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory = load_inventory(Path(args.inventory))
    rows = []
    for node in inventory.nodes:
        inspection = inspect_model_cache(resolve_node_model_cache(inventory, node.name), args.model_id)
        rows.append({"node_name": node.name, "node_rank": node.node_rank, **inspection.to_dict()})
    ready = bool(rows) and all(bool(item["required_files_present"]) for item in rows)
    hashes = {str(item["manifest_hash"]) for item in rows if item["required_files_present"]}
    parity = bool(ready and len(hashes) == 1)
    payload = {
        "model_id": args.model_id,
        "nodes": rows,
        "MODEL_CACHE_MISSING": not ready,
        "MODEL_CACHE_PARITY_PASS": parity,
    }
    print(json.dumps(payload, indent=2))
    return 0 if parity else 2


if __name__ == "__main__":
    raise SystemExit(main())
