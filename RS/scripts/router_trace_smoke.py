#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.runtime import load_model_and_tokenizer
from routesense.topology import load_inventory, resolve_inventory_path, resolve_node_model_cache
from routesense.trace import collect_olmoe_router_trace, summarize_router_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect a real OLMoE router trace smoke.")
    parser.add_argument("--model-id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--text", type=str, default="The history of science is a story of")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "single_gpu_router_trace_smoke"))
    parser.add_argument("--layer-path", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16")
    args = parser.parse_args(argv)

    model_path = args.model_path
    if model_path is None:
        inventory_path = resolve_inventory_path()
        if inventory_path.exists():
            inventory = load_inventory(inventory_path)
            cache = resolve_node_model_cache(inventory, "node1")
            if cache is not None:
                model_path = str(cache / "OLMoE-1B-7B-0924")
    model_path = model_path or "/root/autodl-tmp/models/OLMoE-1B-7B-0924"
    model, tokenizer, _, _, _ = load_model_and_tokenizer(
        model_id=args.model_id,
        model_path=model_path,
        precision=args.precision,
        device_index=0,
    )
    trace = collect_olmoe_router_trace(model, tokenizer, args.text, layer_path=args.layer_path)
    summary = summarize_router_trace(trace)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "router_trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in trace["records"]:
            handle.write(json.dumps(record) + "\n")
    (out / "goal.md").write_text(
        "# Single-GPU Router Trace Smoke\n\nThis run captures a real OLMoE router trace only. It does not perform dispatch, scheduling, or performance comparison.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
