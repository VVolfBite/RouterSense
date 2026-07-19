#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime import run_single_gpu_text_inference
from rs.topology import load_inventory, resolve_inventory_path, resolve_preferred_model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a real OLMoE checkpoint and run one short generation.")
    parser.add_argument("--inventory", type=str, default=None)
    parser.add_argument("--node-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--max-memory-gb", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "single_gpu_olmoe_smoke"))
    args = parser.parse_args(argv)
    model_id = "allenai/OLMoE-1B-7B-0924-Instruct"
    model_path = args.model_path
    if model_path is None:
        inventory_path = Path(args.inventory) if args.inventory else resolve_inventory_path()
        if inventory_path.exists():
            inventory = load_inventory(inventory_path)
            candidate = resolve_preferred_model_path(inventory, model_id, preferred_node_name=args.node_name)
            if candidate is not None:
                model_path = str(candidate)
    model_path = model_path or os.environ.get("RS_MODEL_PATH")
    result = run_single_gpu_text_inference(
        model_id=model_id,
        model_path=model_path,
        prompt="The history of science is a story of",
        max_new_tokens=16,
        precision="bf16",
        device_index=0,
        device_map=args.device_map,
        max_memory_gb=args.max_memory_gb,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
