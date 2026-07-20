#!/usr/bin/env python3
from __future__ import annotations

"""Inspect or materialize the deployment model snapshot for one node."""

import argparse
import json
import os
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("node_name", nargs="?", default="node0")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_DEPLOYMENT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inventory = load_inventory(Path(args.inventory))
    configured = resolve_node_model_cache(inventory, args.node_name)
    before = inspect_model_cache(configured, args.model_id)
    payload: dict[str, object] = {
        "schema_version": "routersense.deploy.model_prefetch.v2",
        "node_name": args.node_name,
        "apply_mode": bool(args.apply),
        "model_id": args.model_id,
        "revision": args.revision or "default",
        "before": before.to_dict(),
        "target_model_path": before.model_path,
        "gpu_runtime_attempted": False,
    }
    if not args.apply:
        payload["status"] = "READY" if before.required_files_present else "DRY_RUN_MISSING"
        print(json.dumps(payload, indent=2))
        return 0

    if not before.required_files_present:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            payload.update({"status": "FAIL", "reason": f"huggingface_hub_unavailable: {type(exc).__name__}: {exc}"})
            print(json.dumps(payload, indent=2))
            return 2
        target = Path(before.model_path)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(args.model_id),
            revision=args.revision,
            local_dir=str(target),
            token=os.environ.get("HF_TOKEN") or None,
            local_files_only=bool(args.local_files_only),
        )
    after = inspect_model_cache(configured, args.model_id)
    payload["after"] = after.to_dict()
    payload["status"] = "PASS" if after.required_files_present else "FAIL"
    print(json.dumps(payload, indent=2))
    return 0 if after.required_files_present else 2


if __name__ == "__main__":
    raise SystemExit(main())
