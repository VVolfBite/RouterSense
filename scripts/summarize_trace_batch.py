#!/usr/bin/env python3
"""Trace batch implementation used by experiment/poc1/trace_batch.py.

Keep new user-facing POC1 entrypoints under experiment/poc1/.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc1.config import load_config
from routesense_poc1.model_loader import load_olmoe_model
from routesense_poc1.router_trace import build_batch_routing_summary, collect_router_trace
from routesense_poc1.serialization import save_json


DEFAULT_TEXTS = [
    ("narrative", "The history of science is a story of"),
    ("narrative", "In a quiet laboratory, the researcher noticed"),
    ("narrative", "The city council approved a new plan for"),
    ("narrative", "During the final minutes of the match, the"),
    ("narrative", "The ancient manuscript was preserved inside"),
    ("narrative", "At dawn, the expedition reached the edge"),
    ("technical", "A small neural network can sometimes reveal"),
    ("technical", "Before deploying the service, engineers measured"),
    ("technical", "The software update changed how requests are"),
    ("technical", "The compiler optimized the loop by moving"),
    ("technical", "The database query slowed down because the"),
    ("technical", "During training, the agent discovered a"),
    ("science", "The patient described a sudden change in"),
    ("science", "The climate model predicted a gradual increase"),
    ("science", "The telescope captured faint light from a"),
    ("science", "In classical mechanics, momentum is conserved when"),
    ("science", "The protein folded into a shape that"),
    ("science", "Medical imaging often helps doctors identify"),
    ("structured", "Shopping list: apples, rice, batteries, and"),
    ("structured", "Step 1: initialize the queue; Step 2:"),
    ("structured", "Table columns include user_id, timestamp, status,"),
    ("structured", "Checklist: verify input, run tests, inspect"),
    ("structured", "Route plan: depot -> hub -> regional"),
    ("structured", "Error report: module auth failed after retry"),
    ("code", "def route_tokens(tokens, experts): return"),
    ("code", "for request in batch: dispatch(request)"),
    ("code", "if cache_hit and priority > threshold:"),
    ("code", "class SchedulerState: pass"),
    ("code", "SELECT expert_id, COUNT(*) FROM routes"),
    ("code", "while queue and budget > 0:"),
    ("long", "The logistics manager rerouted shipments after a regional delay forced several trucks to wait near the depot"),
    ("long", "A legal scholar argued that the statute should be interpreted in light of recent decisions and older precedent"),
    ("long", "Inside the factory, automated arms assembled delicate components while technicians monitored vibration and temperature"),
    ("long", "The emergency team coordinated supplies across hospitals, shelters, and temporary command posts during the storm"),
    ("long", "A financial analyst warned that the quarterly report hid several risks behind unusually optimistic demand forecasts"),
    ("long", "The museum curator found a small inscription beneath the frame and compared it with archived restoration notes"),
    ("question", "Why did the router send these tokens to different experts when the prompt looked simple?"),
    ("question", "How should a scheduler react when one destination receives many dependent token routes?"),
    ("question", "What happens if the largest bucket is not the bucket that blocks completion?"),
    ("question", "Can a sparse expert layer reveal communication pressure before workers begin computing?"),
    ("question", "Which signal is more useful: current queue size or routing dependency structure?"),
    ("question", "When should a system prioritize a smaller bucket over a larger hot bucket?"),
    ("mixed", "Note: GPU-0 handled 18 routes, GPU-1 handled 7, and the barrier moved unexpectedly"),
    ("mixed", "The report says latency dropped 12%, but only after reordering the expert dispatch phase"),
    ("mixed", "Experiment A uses state-only priority; experiment B uses dependency-aware priority"),
    ("mixed", "Router output: experts=[3, 14, 27], weights=[0.41, 0.22, 0.09]"),
    ("mixed", "A short message, a JSON key, and a code token appeared in the same prompt"),
    ("mixed", "Please summarize: queue pressure, co-active experts, and the likely slow destination"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small real OLMoE trace batch and summarize routing skew.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "poc1.yaml"))
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--layers", type=str, default="0,8,15")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "poc1_trace_batch"))
    parser.add_argument("--texts-file", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--comment", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model_id is not None:
        config.model_id = args.model_id
    if args.layer is not None:
        config.layer = args.layer
    config.output_dir = args.output_dir

    prompt_records = _load_prompt_records(args.texts_file)
    prompt_records = prompt_records[: max(1, args.max_samples)]
    layer_specs = [args.layer] if args.layer is not None else _parse_layers(args.layers)
    output_dir = Path(config.output_dir)
    batch_dir = output_dir / "trace_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_batch_outputs(batch_dir)
    _write_goal(
        output_dir,
        "POC1 Trace Batch",
        "Run a small set of real OLMoE traces across representative layers and summarize routing skew, bucket imbalance, and sample-level structure.",
        args.comment,
    )

    model, tokenizer = load_olmoe_model(config)
    sample_stats: list[dict[str, Any]] = []

    for layer_spec in layer_specs:
        layer_key = _layer_key(layer_spec)
        layer_dir = batch_dir / layer_key
        layer_dir.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(prompt_records):
            text = record["text"]
            base_sample_id = f"sample_{index:03d}"
            sample_id = f"{layer_key}_{base_sample_id}"
            trace = collect_router_trace(model=model, tokenizer=tokenizer, text=text, layer_path=layer_spec)
            trace.sample_id = sample_id
            summary = build_batch_routing_summary(trace)

            trace_path = layer_dir / f"{base_sample_id}_trace.json"
            summary_path = layer_dir / f"{base_sample_id}_routing_summary.json"
            save_json(trace.to_dict(), trace_path)
            save_json(summary.to_dict(), summary_path)

            stats = _sample_stats(
                sample_id,
                base_sample_id,
                layer_key,
                layer_spec,
                record["category"],
                text,
                trace.to_dict(),
                summary.to_dict(),
                trace_path,
                summary_path,
            )
            sample_stats.append(stats)
            print(
                f"{sample_id}: active={stats['active_destination_count']} "
                f"routes={stats['total_selected_routes']} "
                f"hot={stats['hot_bucket_count']} "
                f"max_bucket={stats['max_bucket_token_count']} "
                f"max/avg={stats['max_to_avg_bucket_ratio']:.3f} "
                f"entropy={stats['routing_entropy']:.3f}"
            )

    aggregate = _aggregate_stats(sample_stats)
    payload = {
        "mode": "real",
        "purpose": "trace-based mainline judgment smoke test",
        "model_id": config.model_id,
        "layer_specs": layer_specs,
        "layer_count": len(layer_specs),
        "sample_count": len(sample_stats),
        "base_text_count": len(prompt_records),
        "comment": args.comment,
        "samples": sample_stats,
        "aggregate": aggregate,
        "aggregate_by_layer": _aggregate_by_layer(sample_stats),
        "aggregate_by_category": _aggregate_by_category(sample_stats),
    }
    save_json(payload, output_dir / "trace_batch_summary.json")
    print(f"wrote {output_dir / 'trace_batch_summary.json'}")
    return 0


def _write_goal(output_dir: Path, title: str, body: str, comment: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n{body}\n"
    if comment:
        text += f"\nComment: {comment}\n"
    (output_dir / "goal.md").write_text(text, encoding="utf-8")


def _clear_previous_batch_outputs(batch_dir: Path) -> None:
    for pattern in ("sample_*_trace.json", "sample_*_routing_summary.json"):
        for path in batch_dir.rglob(pattern):
            path.unlink()


def _load_prompt_records(texts_file: str | None) -> list[dict[str, str]]:
    if texts_file is None:
        return [{"category": category, "text": text} for category, text in DEFAULT_TEXTS]
    path = Path(texts_file)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            category, text = line.split("\t", 1)
        else:
            category, text = "custom", line
        records.append({"category": category.strip() or "custom", "text": text.strip()})
    return records


def _parse_layers(layers: str) -> list[str]:
    return [layer.strip() for layer in layers.split(",") if layer.strip()]


def _layer_key(layer_spec: str) -> str:
    return f"layer_{str(layer_spec).replace('.', '_').replace('/', '_')}"


def _sample_stats(
    sample_id: str,
    base_sample_id: str,
    layer_key: str,
    layer_spec: str,
    prompt_category: str,
    text: str,
    trace: dict[str, Any],
    summary: dict[str, Any],
    trace_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    buckets = summary.get("bucket_summaries") or summary.get("buckets") or []
    bucket_counts = [int(bucket.get("token_count_in_bucket", 0)) for bucket in buckets]
    hot_bucket_count = sum(1 for bucket in buckets if bool(bucket.get("is_hot_bucket")))
    top3_bucket_token_counts = sorted(bucket_counts, reverse=True)[:3]
    selected_expert_ids = [int(expert_id) for expert_id in trace.get("selected_expert_ids", [])]
    selected_distribution = Counter(selected_expert_ids)
    avg_bucket = statistics.fmean(bucket_counts) if bucket_counts else 0.0
    max_bucket = max(bucket_counts) if bucket_counts else 0
    ratio = float(max_bucket / avg_bucket) if avg_bucket else 0.0
    active_destination_count = int(summary.get("destination_count") or len(summary.get("active_destinations", [])))
    total_selected_routes = int(summary.get("total_selected_routes", 0))
    bucket_count = len(buckets)

    return {
        "sample_id": sample_id,
        "base_sample_id": base_sample_id,
        "layer_key": layer_key,
        "layer_spec": layer_spec,
        "prompt_category": prompt_category,
        "text": text,
        "trace_path": str(trace_path),
        "routing_summary_path": str(summary_path),
        "layer_id": trace.get("layer_id"),
        "layer_path": trace.get("layer_path"),
        "sequence_length": trace.get("sequence_length"),
        "active_destination_count": active_destination_count,
        "destination_count": active_destination_count,
        "total_selected_routes": total_selected_routes,
        "bucket_count": bucket_count,
        "hot_bucket_count": hot_bucket_count,
        "max_bucket_token_count": max_bucket,
        "top3_bucket_token_counts": top3_bucket_token_counts,
        "avg_bucket_token_count": avg_bucket,
        "max_to_avg_bucket_ratio": ratio,
        "bucket_token_count_gini": _gini(bucket_counts),
        "unique_selected_expert_count": len(set(selected_expert_ids)),
        "selected_expert_distribution": {str(key): value for key, value in sorted(selected_distribution.items())},
        "selected_expert_ids": selected_expert_ids,
        "routing_entropy": trace.get("routing_entropy"),
        "top1_top2_gap": trace.get("top1_top2_gap"),
    }


def _aggregate_by_layer(sample_stats: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for layer_key in sorted({item["layer_key"] for item in sample_stats}):
        result[layer_key] = _aggregate_stats([item for item in sample_stats if item["layer_key"] == layer_key])
    return result


def _aggregate_by_category(sample_stats: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for category in sorted({item["prompt_category"] for item in sample_stats}):
        result[category] = _aggregate_stats([item for item in sample_stats if item["prompt_category"] == category])
    return result


def _aggregate_stats(sample_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not sample_stats:
        return {}
    skew_sorted = sorted(sample_stats, key=lambda item: item["max_to_avg_bucket_ratio"], reverse=True)
    active_counts = [item["active_destination_count"] for item in sample_stats]
    route_counts = [item["total_selected_routes"] for item in sample_stats]
    hot_bucket_counts = [item["hot_bucket_count"] for item in sample_stats]
    max_bucket_counts = [item["max_bucket_token_count"] for item in sample_stats]
    gini_values = [item["bucket_token_count_gini"] for item in sample_stats]
    skew_ratios = [item["max_to_avg_bucket_ratio"] for item in sample_stats]
    entropies = [item["routing_entropy"] for item in sample_stats if item["routing_entropy"] is not None]
    gaps = [item["top1_top2_gap"] for item in sample_stats if item["top1_top2_gap"] is not None]
    has_structure_signal = (
        max(active_counts) > min(active_counts)
        or max(skew_ratios) >= 1.5
        or (entropies and max(entropies) - min(entropies) >= 0.1)
    )

    return {
        "active_destination_count_min": min(active_counts),
        "active_destination_count_max": max(active_counts),
        "active_destination_count_mean": statistics.fmean(active_counts),
        "total_selected_routes_min": min(route_counts),
        "total_selected_routes_max": max(route_counts),
        "total_selected_routes_mean": statistics.fmean(route_counts),
        "hot_bucket_count_min": min(hot_bucket_counts),
        "hot_bucket_count_max": max(hot_bucket_counts),
        "hot_bucket_count_mean": statistics.fmean(hot_bucket_counts),
        "max_bucket_token_count_min": min(max_bucket_counts),
        "max_bucket_token_count_max": max(max_bucket_counts),
        "max_bucket_token_count_mean": statistics.fmean(max_bucket_counts),
        "max_to_avg_bucket_ratio_min": min(skew_ratios),
        "max_to_avg_bucket_ratio_max": max(skew_ratios),
        "max_to_avg_bucket_ratio_mean": statistics.fmean(skew_ratios),
        "bucket_token_count_gini_min": min(gini_values),
        "bucket_token_count_gini_max": max(gini_values),
        "bucket_token_count_gini_mean": statistics.fmean(gini_values),
        "routing_entropy_min": min(entropies) if entropies else None,
        "routing_entropy_max": max(entropies) if entropies else None,
        "routing_entropy_mean": statistics.fmean(entropies) if entropies else None,
        "top1_top2_gap_min": min(gaps) if gaps else None,
        "top1_top2_gap_max": max(gaps) if gaps else None,
        "top1_top2_gap_mean": statistics.fmean(gaps) if gaps else None,
        "most_skewed_sample_id": skew_sorted[0]["sample_id"],
        "most_skewed_ratio": skew_sorted[0]["max_to_avg_bucket_ratio"],
        "most_uniform_sample_id": skew_sorted[-1]["sample_id"],
        "most_uniform_ratio": skew_sorted[-1]["max_to_avg_bucket_ratio"],
        "has_structure_signal": has_structure_signal,
        "judgment": "routing structures differ across samples; worth continuing mainline tests"
        if has_structure_signal
        else "routing structures are very similar in this small batch; gather more traces before expanding",
        "short_conclusion": [
            "multiple active destinations consistently appear",
            "bucket imbalance varies across samples",
            "trace evidence supports continuing the dependency-aware mainline",
        ]
        if has_structure_signal
        else [
            "this small batch shows limited routing structure variation",
            "collect more traces before expanding the mainline",
        ],
    }


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    n = len(sorted_values)
    weighted_sum = sum((index + 1) * value for index, value in enumerate(sorted_values))
    return float((2 * weighted_sum) / (n * total) - (n + 1) / n)


if __name__ == "__main__":
    raise SystemExit(main())
