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
    parser = argparse.ArgumentParser(description="Verify one inventory node's local model snapshot.")
    parser.add_argument("inventory")
    parser.add_argument("node_name", nargs="?", default="node0")
    parser.add_argument("--model-id", default=DEFAULT_DEPLOYMENT_MODEL_ID)
    parser.add_argument("--print-model-path", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory = load_inventory(Path(args.inventory))
    cache = resolve_node_model_cache(inventory, args.node_name)
    inspection = inspect_model_cache(cache, args.model_id)
    payload = {
        "node_name": args.node_name,
        **inspection.to_dict(),
        "gpu_runtime_attempted": False,
    }
    if args.print_model_path:
        print(inspection.model_path)
    else:
        print(json.dumps(payload, indent=2))
    return 0 if inspection.required_files_present else 2


if __name__ == "__main__":
    raise SystemExit(main())
