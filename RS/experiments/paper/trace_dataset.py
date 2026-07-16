from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rs.core.hashing import stable_hash_dict

from .contracts import RecordMetadata, TraceSample


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_replay_fixture(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "path": str(path),
        "name": path.stem,
        "num_gpus": int(payload["num_gpus"]) if "num_gpus" in payload else len(payload["p0_dispatch_matrix"]),
        "p0_dispatch_matrix": tuple(tuple(int(v) for v in row) for row in payload["p0_dispatch_matrix"]),
        "p1_return_matrix": tuple(tuple(int(v) for v in row) for row in payload["p1_return_matrix"]),
        "p2_next_dispatch_matrix": tuple(tuple(int(v) for v in row) for row in payload["p2_next_dispatch_matrix"]),
        "metadata": dict(payload.get("metadata", {})),
    }


def replay_fixture_to_trace_sample(
    fixture: dict[str, Any],
    *,
    model_id: str,
    model_revision: str,
    metadata: RecordMetadata,
) -> TraceSample:
    counts = fixture["p0_dispatch_matrix"]
    layer_id = str(fixture["metadata"].get("layer_id", fixture["name"]))
    trace_sample_id = f"{fixture['name']}:{layer_id}"
    return TraceSample(
        schema_version="paper.trace_sample.v2",
        model_id=str(model_id),
        model_revision=str(model_revision),
        prompt_id=str(fixture["name"]),
        batch_id="offline-replay",
        sequence_length=int(sum(sum(row) for row in counts)),
        layer_id=layer_id,
        num_experts=int(fixture["num_gpus"]),
        top_k=0,
        router_logits_digest=None,
        selected_experts_digest=stable_hash_dict({"p0": [list(row) for row in counts]}),
        routing_weights_digest=None,
        compact_route_counts=counts,
        capture_timestamp="offline_fixture",
        metadata=metadata,
        source_kind="replay_fixture_proxy",
        trace_sample_id=trace_sample_id,
        trace_bundle_path=str(fixture["path"]),
    )


def discover_replay_fixtures(fixture_dir: Path) -> list[Path]:
    return sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda item: item.name)


def load_trace_bundle(bundle_dir: Path) -> dict[str, Any]:
    trace_path = bundle_dir / "trace.jsonl"
    summary_path = bundle_dir / "summary.json"
    architecture_path = bundle_dir / "architecture_probe.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"trace bundle missing trace.jsonl: {bundle_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"trace bundle missing summary.json: {bundle_dir}")
    if not architecture_path.exists():
        raise FileNotFoundError(f"trace bundle missing architecture_probe.json: {bundle_dir}")
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "bundle_dir": str(bundle_dir),
        "summary": load_json(summary_path),
        "architecture_probe": load_json(architecture_path),
        "run_manifest": load_json(bundle_dir / "run_manifest.json") if (bundle_dir / "run_manifest.json").exists() else {},
        "records": records,
    }


def _layer_num_experts(architecture_probe: dict[str, Any], layer_id: str) -> int:
    for row in architecture_probe.get("layers", []):
        if str(row.get("layer_index")) == str(layer_id):
            shape = row.get("gate_weight_shape")
            if isinstance(shape, list) and shape:
                return int(shape[0])
    return int(architecture_probe.get("moe_layer_count", 0) or 0)


def trace_bundle_to_trace_samples(bundle_dir: Path, *, metadata: RecordMetadata) -> list[TraceSample]:
    payload = load_trace_bundle(bundle_dir)
    by_sample_layer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in payload["records"]:
        by_sample_layer[(str(record["sample_id"]), str(record["layer_id"]))].append(record)
    samples: list[TraceSample] = []
    for (sample_id, layer_id), records in sorted(by_sample_layer.items()):
        counts_by_expert: dict[int, int] = defaultdict(int)
        weights: list[float] = []
        token_positions: set[int] = set()
        topk = 0
        for row in records:
            counts_by_expert[int(row["expert_id"])] += 1
            weights.append(float(row["routing_weight"]))
            token_positions.add(int(row["token_position"]))
            topk = max(topk, int(row.get("topk", 0) or 0))
        num_experts = _layer_num_experts(payload["architecture_probe"], layer_id)
        compact = (tuple(int(counts_by_expert.get(expert_id, 0)) for expert_id in range(num_experts)),)
        samples.append(
            TraceSample(
                schema_version="paper.trace_sample.v2",
                model_id=str(metadata.model_id),
                model_revision=str(metadata.model_revision),
                prompt_id=str(sample_id),
                batch_id="real-trace",
                sequence_length=len(token_positions),
                layer_id=str(layer_id),
                num_experts=int(num_experts),
                top_k=int(topk),
                router_logits_digest=None,
                selected_experts_digest=stable_hash_dict({"records": [(int(r["token_position"]), int(r["expert_id"])) for r in records]}),
                routing_weights_digest=stable_hash_dict({"weights": [float(item) for item in weights]}),
                compact_route_counts=compact,
                capture_timestamp="trace_bundle",
                metadata=metadata,
                source_kind="real_router_trace",
                trace_sample_id=f"{sample_id}:{layer_id}",
                trace_bundle_path=str(bundle_dir),
            )
        )
    return samples
