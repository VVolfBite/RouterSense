#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # type: ignore

from routesense.evaluation import collect_environment_snapshot, write_json
from routesense.runtime import load_model_and_tokenizer
from routesense.trace import collect_qwen_moe_architecture_probe, collect_qwen_moe_full_sequence_trace


def _load_prompt_entries(path: Path, limit: int | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            entries.append(payload)
            if limit is not None and len(entries) >= limit:
                break
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect a full-sequence Qwen-MoE router trace for POC-line1.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/models/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--prompts-path", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--request-id", type=str, default="qwen-req-0")
    parser.add_argument("--sample-id", type=str, default="qwen-sample-0")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "full_sequence_trace_qwen"),
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
    if bool(args.text) == bool(args.prompts_path):
        raise SystemExit("exactly one of --text or --prompts-path must be provided")

    architecture_probe = collect_qwen_moe_architecture_probe(model)
    traces: list[dict[str, Any]] = []
    if args.text is not None:
        traces.append(
            collect_qwen_moe_full_sequence_trace(
                model,
                tokenizer,
                args.text,
                request_id=args.request_id,
                sample_id=args.sample_id,
                save_auxiliary_dir=Path(args.output_dir) / "auxiliary",
            )
        )
    else:
        prompt_entries = _load_prompt_entries(Path(args.prompts_path), args.num_prompts)
        for index, entry in enumerate(prompt_entries):
            text = str(entry["text"])
            document_id = entry.get("document_id", index)
            traces.append(
                collect_qwen_moe_full_sequence_trace(
                    model,
                    tokenizer,
                    text,
                    request_id=f"{args.request_id}-{index}",
                    sample_id=f"prompt-{document_id}",
                    save_auxiliary_dir=Path(args.output_dir) / "auxiliary",
                )
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
            "trace_backend": "qwen_moe",
            "request_id": args.request_id,
            "sample_id": args.sample_id,
            "prompts_path": args.prompts_path,
            "num_prompts": args.num_prompts,
        },
    )
    write_json(out / "environment.json", collect_environment_snapshot())
    write_json(out / "architecture_probe.json", architecture_probe)
    merged_hidden_states: dict[str, dict[int, torch.Tensor]] = {}
    merged_gate_weights: dict[str, dict[int, torch.Tensor]] = {}
    with (out / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            for record in trace["records"]:
                handle.write(json.dumps(record) + "\n")
            merged_hidden_states[trace["summary"]["sample_id"]] = trace["hidden_states"]
            merged_gate_weights[trace["summary"]["sample_id"]] = trace["gate_weights"]
    torch.save(merged_hidden_states, out / "hidden_states.pt")
    torch.save(merged_gate_weights, out / "gate_weights.pt")
    combined_summary = {
        "trace_backend": "qwen_moe",
        "trace_count": len(traces),
        "sample_ids": [trace["summary"]["sample_id"] for trace in traces],
        "request_ids": [trace["summary"]["request_id"] for trace in traces],
        "moe_layer_count": traces[0]["summary"]["moe_layer_count"] if traces else 0,
        "moe_layer_ids": traces[0]["summary"].get("moe_layer_ids", []) if traces else [],
        "topk": traces[0]["summary"]["topk"] if traces else 0,
        "token_count_per_sample": {trace["summary"]["sample_id"]: trace["summary"]["token_count"] for trace in traces},
        "record_count": sum(int(trace["summary"]["record_count"]) for trace in traces),
        "hidden_states_path": str(out / "hidden_states.pt"),
        "gate_weights_path": str(out / "gate_weights.pt"),
    }
    write_json(out / "summary.json", combined_summary)
    print(json.dumps(combined_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
