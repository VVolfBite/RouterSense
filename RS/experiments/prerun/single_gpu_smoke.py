#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.runtime import run_single_gpu_text_inference
from routesense.topology import load_inventory, resolve_inventory_path, resolve_node_model_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a real OLMoE checkpoint and run one short generation.")
    parser.add_argument("--inventory", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "single_gpu_olmoe_smoke"))
    args = parser.parse_args(argv)
    model_path = args.model_path
    if model_path is None:
        inventory_path = Path(args.inventory) if args.inventory else resolve_inventory_path()
        if inventory_path.exists():
            inventory = load_inventory(inventory_path)
            cache = resolve_node_model_cache(inventory, "node1")
            if cache is not None:
                model_path = str(cache / "OLMoE-1B-7B-0924")
    model_path = model_path or os.environ.get("RS_MODEL_PATH") or "/root/autodl-tmp/models/OLMoE-1B-7B-0924"
    result = run_single_gpu_text_inference(
        model_id="allenai/OLMoE-1B-7B-0924-Instruct",
        model_path=model_path,
        prompt="The history of science is a story of",
        max_new_tokens=16,
        precision="bf16",
        device_index=0,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
