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

from routesense.evaluation import collect_environment_snapshot, write_json
from routesense.runtime import load_model_and_tokenizer
from routesense.trace import collect_full_sequence_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect a full-sequence OLMoE router trace for POC-line1.")
    parser.add_argument("--model-id", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--request-id", type=str, default="req-0")
    parser.add_argument("--sample-id", type=str, default="sample-0")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "full_sequence_trace"),
    )
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args(argv)

    model, tokenizer, _, _, source = load_model_and_tokenizer(
        model_id=args.model_id,
        model_path=args.model_path,
        precision=args.precision,
        device_index=args.device_index,
    )
    trace = collect_full_sequence_trace(
        model,
        tokenizer,
        args.text,
        request_id=args.request_id,
        sample_id=args.sample_id,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "config.json",
        {
            "model_id": args.model_id,
            "model_path": args.model_path,
            "model_source": source,
            "precision": args.precision,
            "request_id": args.request_id,
            "sample_id": args.sample_id,
        },
    )
    write_json(out / "environment.json", collect_environment_snapshot())
    with (out / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in trace["records"]:
            handle.write(json.dumps(record) + "\n")
    write_json(out / "summary.json", trace["summary"])
    print(json.dumps(trace["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
