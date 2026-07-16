from __future__ import annotations

import json
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
        schema_version="paper.trace_sample.v1",
        model_id=str(model_id),
        model_revision=str(model_revision),
        prompt_id=str(fixture["name"]),
        batch_id="offline-replay",
        sequence_length=int(sum(sum(row) for row in counts)),
        layer_id=layer_id,
        num_experts=int(fixture["num_gpus"]),
        top_k=0,
        router_logits_digest="not_available_from_replay_fixture",
        selected_experts_digest=stable_hash_dict({"p0": [list(row) for row in counts]}),
        routing_weights_digest="not_available_from_replay_fixture",
        compact_route_counts=counts,
        capture_timestamp="offline_fixture",
        metadata=metadata,
        source_kind="replay_fixture_proxy",
        trace_sample_id=trace_sample_id,
    )


def discover_replay_fixtures(fixture_dir: Path) -> list[Path]:
    return sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda item: item.name)
