from __future__ import annotations

"""Validated FATE-P2 artifacts for the Current-P12 runtime.

The simulator deliberately does not infer FATE from future routing truth.  A
FATE prediction must be produced by a real cross-layer gate predictor (online
or offline) and embedded in the current P0 trace-window metadata.  Missing or
malformed artifacts fail closed instead of falling back to the last-value
predictor.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from rs_sim.scheduler.stable import stable_digest

FATE_P2_ARTIFACT_SCHEMA = "ROUTERSENSE_FATE_P2_ARTIFACT"
_FATE_P2_ARTIFACT_SCHEMAS = {
    FATE_P2_ARTIFACT_SCHEMA,
    "ROUTERSENSE_FATE_P2_ARTIFACT_V1",
}
FATE_METADATA_KEY = "fate_p2_prediction"


class FateP2ArtifactError(ValueError):
    pass


def _square_nonnegative_matrix(value: Any, *, world_size: int, name: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != int(world_size):
        raise FateP2ArtifactError(f"{name} must contain world_size rows")
    rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != int(world_size):
            raise FateP2ArtifactError(f"{name}[{row_index}] must contain world_size values")
        normalized = tuple(int(item) for item in row)
        if any(item < 0 for item in normalized):
            raise FateP2ArtifactError(f"{name} must be non-negative")
        rows.append(normalized)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class FateP2Artifact:
    predictor_id: str
    source_layer_id: int
    target_layer_id: int
    confidence_ppm: int
    routing_rows: tuple[tuple[int, ...], ...] | None
    payload_matrix: tuple[tuple[int, ...], ...] | None
    estimator_kind: str
    artifact_digest: str
    source_artifact_digest: str

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        world_size: int,
        source_layer_id: int,
        target_layer_id: int,
    ) -> "FateP2Artifact":
        raw = metadata.get(FATE_METADATA_KEY)
        if not isinstance(raw, Mapping):
            raise FateP2ArtifactError(
                f"Current P0 metadata is missing {FATE_METADATA_KEY!r}; FATE_P2 never falls back"
            )
        schema = str(raw.get("schema_version", ""))
        if schema not in _FATE_P2_ARTIFACT_SCHEMAS:
            raise FateP2ArtifactError(f"unsupported FATE artifact schema {schema!r}")
        predictor_id = str(raw.get("predictor_id", "")).strip()
        if not predictor_id:
            raise FateP2ArtifactError("FATE predictor_id must be non-empty")
        source = int(raw.get("source_layer_id", -1))
        target = int(raw.get("target_layer_id", -1))
        if source != int(source_layer_id) or target != int(target_layer_id):
            raise FateP2ArtifactError(
                f"FATE layer identity mismatch: artifact={source}->{target}, expected={source_layer_id}->{target_layer_id}"
            )
        confidence_ppm = int(raw.get("confidence_ppm", -1))
        if not 0 <= confidence_ppm <= 1_000_000:
            raise FateP2ArtifactError("FATE confidence_ppm must be in [0, 1000000]")
        routing = raw.get("routing_rows")
        payload = raw.get("payload_matrix")
        if (routing is None) == (payload is None):
            raise FateP2ArtifactError("FATE artifact must contain exactly one of routing_rows or payload_matrix")
        routing_rows = None if routing is None else _square_nonnegative_matrix(
            routing, world_size=world_size, name="routing_rows"
        )
        payload_matrix = None if payload is None else _square_nonnegative_matrix(
            payload, world_size=world_size, name="payload_matrix"
        )
        estimator_kind = str(raw.get("estimator_kind", "")).strip() or (
            "RANK_ROUTING_ROWS" if routing_rows is not None else "PAYLOAD_MATRIX"
        )
        source_artifact_digest = str(raw.get("source_artifact_digest", "")).strip()
        source_semantic = {
            "schema_version": schema,
            "predictor_id": predictor_id,
            "source_layer_id": source,
            "target_layer_id": target,
            "confidence_ppm": confidence_ppm,
            "routing_rows": routing_rows,
            "payload_matrix": payload_matrix,
            "estimator_kind": estimator_kind,
            "source_artifact_digest": source_artifact_digest,
        }
        source_computed = stable_digest(source_semantic)
        declared = str(raw.get("artifact_digest", "")).strip()
        if declared and declared != source_computed:
            raise FateP2ArtifactError("FATE artifact_digest does not match artifact content")
        canonical_semantic = dict(source_semantic)
        canonical_semantic["schema_version"] = FATE_P2_ARTIFACT_SCHEMA
        computed = stable_digest(canonical_semantic)
        return cls(
            predictor_id=predictor_id,
            source_layer_id=source,
            target_layer_id=target,
            confidence_ppm=confidence_ppm,
            routing_rows=routing_rows,
            payload_matrix=payload_matrix,
            estimator_kind=estimator_kind,
            artifact_digest=computed,
            source_artifact_digest=source_artifact_digest,
        )

    def payload_bytes(self, payload_spec: Any) -> tuple[tuple[int, ...], ...]:
        if self.payload_matrix is not None:
            return self.payload_matrix
        assert self.routing_rows is not None
        return tuple(
            tuple(int(payload_spec.edge_payload_bytes(int(rows))) for rows in row)
            for row in self.routing_rows
        )


def canonical_fate_metadata(
    *,
    predictor_id: str,
    source_layer_id: int,
    target_layer_id: int,
    confidence_ppm: int,
    routing_rows: tuple[tuple[int, ...], ...] | None = None,
    payload_matrix: tuple[tuple[int, ...], ...] | None = None,
    estimator_kind: str = "",
    source_artifact_digest: str = "",
) -> dict[str, Any]:
    if (routing_rows is None) == (payload_matrix is None):
        raise FateP2ArtifactError("provide exactly one of routing_rows or payload_matrix")
    body: dict[str, Any] = {
        "schema_version": FATE_P2_ARTIFACT_SCHEMA,
        "predictor_id": str(predictor_id),
        "source_layer_id": int(source_layer_id),
        "target_layer_id": int(target_layer_id),
        "confidence_ppm": int(confidence_ppm),
        "routing_rows": routing_rows,
        "payload_matrix": payload_matrix,
        "estimator_kind": str(estimator_kind) or (
            "RANK_ROUTING_ROWS" if routing_rows is not None else "PAYLOAD_MATRIX"
        ),
        "source_artifact_digest": str(source_artifact_digest),
    }
    semantic = dict(body)
    body["artifact_digest"] = stable_digest(semantic)
    return body


__all__ = [
    "FATE_METADATA_KEY",
    "FATE_P2_ARTIFACT_SCHEMA",
    "FateP2Artifact",
    "FateP2ArtifactError",
    "canonical_fate_metadata",
]
