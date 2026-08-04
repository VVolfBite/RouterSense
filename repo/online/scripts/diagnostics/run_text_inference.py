#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime import run_single_gpu_text_inference
from rs.topology import load_inventory, resolve_inventory_path, resolve_preferred_model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real single-GPU OLMoE text inference smoke.")
    parser.add_argument("--model-id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--inventory", type=str, default=None)
    parser.add_argument("--node-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="The history of science is a story of")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-map", type=str, default=None)
    parser.add_argument("--max-memory-gb", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "single_gpu_text_infer"))
    args = parser.parse_args(argv)
    model_path = args.model_path
    if model_path is None:
        inventory_path = Path(args.inventory) if args.inventory else resolve_inventory_path()
        if inventory_path.exists():
            inventory = load_inventory(inventory_path)
            candidate = resolve_preferred_model_path(inventory, args.model_id, preferred_node_name=args.node_name)
            if candidate is not None:
                model_path = str(candidate)
    model_path = model_path or os.environ.get("RS_MODEL_PATH")

    result = run_single_gpu_text_inference(
        model_id=args.model_id,
        model_path=model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
        device_index=args.device_index,
        revision=args.revision,
        device_map=args.device_map,
        max_memory_gb=args.max_memory_gb,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
