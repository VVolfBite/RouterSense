#!/usr/bin/env python3
"""Formal single-GPU router-trace collection entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import collect_environment_snapshot, write_json
from rs.core.experiment_config import RunConfig, load_run_config
from rs.runtime import load_model_and_tokenizer
from rs.runtime.offline.trace.olmoe import collect_full_sequence_trace, collect_moe_architecture_probe


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args(argv)


def _load_prompt_entries(path: Path, limit: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and "prompts" in payload:
        entries = [{"text": item} if isinstance(item, str) else dict(item) for item in payload["prompts"]]
    else:
        raise ValueError(f"unsupported prompt schema in {path}")
    if limit is not None:
        entries = entries[:limit]
    return entries


def _resolve_model_path(config: RunConfig) -> str | None:
    return config.model.local_path or None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_run_config(
        config_path=args.config,
        overrides=list(args.override),
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    out = Path(config.artifact.output_root) / config.run.name
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer, _, _, source = load_model_and_tokenizer(
        model_id=config.model.model_id,
        model_path=_resolve_model_path(config),
        precision=config.runtime.precision,
        device_index=config.model.device_index,
    )
    architecture_probe = collect_moe_architecture_probe(model)
    prompt_entries = _load_prompt_entries(Path(config.workload.prompts), config.workload.num_prompts)

    traces: list[dict[str, Any]] = []
    for index, entry in enumerate(prompt_entries):
        text = str(entry["text"])
        document_id = entry.get("document_id", index)
        traces.append(
            collect_full_sequence_trace(
                model,
                tokenizer,
                text,
                request_id=f"{config.run.name}-req-{index}",
                sample_id=f"sample-{document_id}",
                save_auxiliary_dir=out / "auxiliary",
            )
        )

    write_json(
        out / "run_manifest.json",
        {
            "run_id": config.run.name,
            "run_kind": config.run.kind,
            "model_family": "olmoe",
            "model_revision": config.model.model_id,
            "model_source": source,
            "precision": config.runtime.precision,
            "device": f"cuda:{config.model.device_index}" if torch.cuda.is_available() else "cpu",
            "workload_hash": config.workload.prompts,
            "trace_schema_version": "v1",
            "source_config_path": config.source_config_path,
        },
    )
    write_json(out / "trace_schema.json", {"version": "v1", "record_type": "router_trace"})
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
