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

from routesense.runtime import run_single_gpu_text_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real single-GPU OLMoE text inference smoke.")
    parser.add_argument("--model-id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/models/OLMoE-1B-7B-0924")
    parser.add_argument("--prompt", type=str, default="The history of science is a story of")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "single_gpu_text_infer"))
    args = parser.parse_args(argv)

    result = run_single_gpu_text_inference(
        model_id=args.model_id,
        model_path=args.model_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
        device_index=args.device_index,
        revision=args.revision,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

