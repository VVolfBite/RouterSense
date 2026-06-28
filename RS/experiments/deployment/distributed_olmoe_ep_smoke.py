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

from routesense.runtime import load_model_and_tokenizer, run_single_gpu_text_inference
from routesense.trace import collect_olmoe_router_trace, summarize_router_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0C distributed OLMoE EP smoke.")
    parser.add_argument("--model-id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--inventory", type=str, default=str(ROOT / "deploy" / "inventory" / "hosts.local.yaml"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="The history of science is a story of")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "phase0c_distributed_ep_smoke"))
    args = parser.parse_args(argv)
    model_path = args.model_path
    if model_path is None:
        model_path = os.environ.get("RS_MODEL_PATH")
    model, tokenizer, _, _, _ = load_model_and_tokenizer(
        model_id=args.model_id,
        model_path=model_path,
        precision=args.precision,
        device_index=args.device_index,
    )
    trace = collect_olmoe_router_trace(model, tokenizer, args.prompt)
    summary = summarize_router_trace(trace)
    reference = run_single_gpu_text_inference(
        model_id=args.model_id,
        model_path=model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
        device_index=args.device_index,
    )
    payload = {
        "inventory": args.inventory,
        "trace_summary": summary,
        "reference": reference.to_dict(),
        "status": "PLACEHOLDER_PHASE0C_ENTRYPOINT_REPLACED_WITH_SINGLE_GPU_REFERENCE",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "distributed_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
