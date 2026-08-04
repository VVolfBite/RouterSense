from __future__ import annotations

"""Import externally produced FATE rank-traffic artifacts into trace fixtures."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from rs_sim.scheduler.prediction.fate_p2 import canonical_fate_metadata
from rs_sim.trace.schema.model import TraceValidationError

FATE_BUNDLE_SCHEMA = "ROUTERSENSE_FATE_P2_BUNDLE"


def canonical_fate_record_digest(record: Mapping[str, Any]) -> str:
    """Digest one raw FATE record excluding its self-referential digest field."""

    payload = {str(key): value for key, value in record.items() if str(key) != "record_digest"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_fate_record_digest(record: Mapping[str, Any], *, context: str) -> None:
    supplied = str(record.get("record_digest", "")).strip()
    if not supplied:
        raise TraceValidationError(f"{context} missing record_digest")
    expected = canonical_fate_record_digest(record)
    if supplied != expected:
        raise TraceValidationError(
            f"{context} record_digest mismatch: supplied={supplied} expected={expected}"
        )


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_fate_bundle(path: Path | str) -> tuple[dict[tuple[str, int, int], dict[str, Any]], str]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TraceValidationError(f"FATE artifact bundle not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise TraceValidationError(f"invalid FATE artifact bundle {source}: {exc}") from exc
    if not isinstance(payload, Mapping) or str(payload.get("schema_version", "")) != FATE_BUNDLE_SCHEMA:
        raise TraceValidationError(f"FATE bundle schema must be {FATE_BUNDLE_SCHEMA}")
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise TraceValidationError("FATE bundle predictions must be a non-empty array")
    source_digest = _file_digest(source)
    output: dict[tuple[str, int, int], dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TraceValidationError(f"FATE prediction[{index}] must be an object")
        sample_id = str(raw.get("sample_id", "")).strip()
        if not sample_id:
            raise TraceValidationError(f"FATE prediction[{index}] missing sample_id")
        source_layer = int(raw.get("source_layer_id", -1))
        target_layer = int(raw.get("target_layer_id", -1))
        if source_layer < 0 or target_layer != source_layer + 1:
            raise TraceValidationError(f"FATE prediction[{index}] must target the next consecutive layer")
        routing = raw.get("routing_rows")
        payload_matrix = raw.get("payload_matrix")
        if routing is not None:
            routing = tuple(tuple(int(item) for item in row) for row in routing)
        if payload_matrix is not None:
            payload_matrix = tuple(tuple(int(item) for item in row) for row in payload_matrix)
        metadata = canonical_fate_metadata(
            predictor_id=str(raw.get("predictor_id", "fate_cross_layer_gate_v1")),
            source_layer_id=source_layer,
            target_layer_id=target_layer,
            confidence_ppm=int(raw.get("confidence_ppm", 0)),
            routing_rows=routing,
            payload_matrix=payload_matrix,
            estimator_kind=str(raw.get("estimator_kind", "")),
            source_artifact_digest=str(raw.get("source_artifact_digest", source_digest)),
        )
        key = (sample_id, source_layer, target_layer)
        if key in output:
            raise TraceValidationError(f"duplicate FATE prediction key {key}")
        output[key] = metadata
    return output, source_digest


__all__ = [
    "FATE_BUNDLE_SCHEMA", "canonical_fate_record_digest",
    "load_fate_bundle", "validate_fate_record_digest",
]
