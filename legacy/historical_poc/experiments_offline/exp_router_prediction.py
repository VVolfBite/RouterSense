#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from _bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.offline.router_prediction import build_router_prediction_result
from rs.runtime import load_model_and_tokenizer
from rs.trace import collect_full_sequence_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline single-GPU router prediction trace collection.")
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "offline" / "router_prediction"))
    args = parser.parse_args(argv)

    run_id = f"offline-router-{uuid.uuid4().hex[:12]}"
    model, tokenizer, resolved_revision, resolved_device, dtype = load_model_and_tokenizer(
        model_id=args.model,
        model_path=args.model_path,
        precision=args.precision,
        device_index=args.device_index,
    )
    trace = collect_full_sequence_trace(
        model,
        tokenizer,
        args.prompt,
        request_id=f"{run_id}-request-0",
        sample_id=f"{run_id}-sample-0",
    )
    result = build_router_prediction_result(
        run_id=run_id,
        extra={
            "model": args.model,
            "model_revision": resolved_revision,
            "device": resolved_device,
            "dtype": dtype,
            "trace_summary": trace["summary"],
            "record_count": len(trace.get("records", [])),
        },
    )
    payload = {
        **result,
        "trace": {
            "summary": trace["summary"],
            "records": trace["records"],
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
