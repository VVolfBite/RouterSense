from __future__ import annotations

"""Deployable traffic-bridge predictor.

The online predictor never trains.  It either copies the current dispatch as a
cheap structural bridge or loads a frozen affine artifact produced offline.
This is intentionally separate from faithful FATE, which predicts experts from
cross-layer gate information.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rs.core.contracts import (
    PredictionContext,
    PredictionHint,
    PredictionResult,
    TrafficHistoryContext,
)
from rs.prediction.api import PredictorSpec


@dataclass(frozen=True)
class BridgeArtifact:
    schema_version: str
    artifact_id: str
    world_size: int
    max_layer: int
    coefficients: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]
    confidence: float
    training_corpus_digest: str | None = None
    model_name: str | None = None

    @staticmethod
    def from_mapping(value: Mapping[str, object]) -> "BridgeArtifact":
        return BridgeArtifact(
            schema_version=str(value.get("schema_version", "routersense.bridge.affine.v1")),
            artifact_id=str(value.get("artifact_id", "")),
            world_size=int(value.get("world_size", 0)),
            max_layer=int(value.get("max_layer", 0)),
            coefficients=tuple(tuple(float(x) for x in row) for row in value.get("coefficients", ())),
            intercept=tuple(float(x) for x in value.get("intercept", ())),
            confidence=float(value.get("confidence", 0.0)),
            training_corpus_digest=(None if value.get("training_corpus_digest") is None else str(value["training_corpus_digest"])),
            model_name=(None if value.get("model_name") is None else str(value["model_name"])),
        )

    @staticmethod
    def load(path: str | Path) -> "BridgeArtifact":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("bridge artifact must be a JSON object")
        return BridgeArtifact.from_mapping(payload)

    def validate(self) -> None:
        if self.schema_version != "routersense.bridge.affine.v1":
            raise ValueError(f"unsupported bridge artifact schema {self.schema_version!r}")
        if not self.artifact_id or self.world_size <= 0 or self.max_layer < 0:
            raise ValueError("bridge artifact identity/world/max_layer invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("bridge confidence outside [0, 1]")
        n = self.world_size
        feature_count = n * n + 2 * n + 3
        output_count = n * n
        coef = np.asarray(self.coefficients, dtype=np.float64)
        intercept = np.asarray(self.intercept, dtype=np.float64)
        if coef.shape != (output_count, feature_count):
            raise ValueError(f"bridge coefficients shape {coef.shape} != {(output_count, feature_count)}")
        if intercept.shape != (output_count,):
            raise ValueError("bridge intercept shape mismatch")
        if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
            raise ValueError("bridge artifact contains non-finite values")

    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "world_size": self.world_size,
            "max_layer": self.max_layer,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "confidence": self.confidence,
            "training_corpus_digest": self.training_corpus_digest,
            "model_name": self.model_name,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class BridgeFormalConfig:
    mode: str = "copy_current"
    confidence: float = 0.25
    artifact_path: str | None = None
    artifact: Mapping[str, object] | None = None

    def validate(self) -> None:
        if self.mode not in {"copy_current", "frozen_affine"}:
            raise ValueError("bridge mode must be copy_current or frozen_affine")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("bridge confidence outside [0, 1]")
        if self.mode == "frozen_affine" and (self.artifact_path is None) == (self.artifact is None):
            raise ValueError("frozen_affine requires exactly one of artifact_path or artifact")


def _rows(matrix: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in matrix)


def _source_layer(identity) -> int | None:
    value = getattr(identity, "source_layer_id", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _feature(matrix: np.ndarray, *, source_layer: int, max_layer: int) -> np.ndarray:
    rows = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(rows, 0.0)
    total = max(float(rows.sum()), 1.0)
    return np.concatenate(
        [
            (rows / total).ravel(),
            rows.sum(axis=1) / total,
            rows.sum(axis=0) / total,
            np.asarray(
                [math.log1p(total) / 10.0, float(source_layer) / max(float(max_layer), 1.0), 1.0],
                dtype=np.float64,
            ),
        ]
    )


def _round_remote(raw: np.ndarray, total: int) -> np.ndarray:
    value = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    np.fill_diagonal(value, 0.0)
    if total <= 0 or float(value.sum()) <= 0.0:
        return np.zeros(value.shape, dtype=np.int32)
    scaled = value * (float(total) / float(value.sum()))
    base = np.floor(scaled).astype(np.int32)
    remainder = int(total - int(base.sum()))
    if remainder > 0:
        order = np.argsort(-(scaled - base).ravel(), kind="stable")
        for flat in order:
            i, j = np.unravel_index(int(flat), base.shape)
            if i == j:
                continue
            base[i, j] += 1
            remainder -= 1
            if remainder == 0:
                break
    np.fill_diagonal(base, 0)
    return base


class BridgeFormalPredictor:
    def __init__(self, config: BridgeFormalConfig | None = None) -> None:
        self.config = config or BridgeFormalConfig()
        self.config.validate()
        self._artifact: BridgeArtifact | None = None
        if self.config.mode == "frozen_affine":
            self._artifact = (
                BridgeArtifact.load(self.config.artifact_path)
                if self.config.artifact_path is not None
                else BridgeArtifact.from_mapping(self.config.artifact or {})
            )
            self._artifact.validate()

    @property
    def predictor_id(self) -> str:
        return "bridge_frozen_affine" if self._artifact is not None else "bridge_copy_current"

    def predict(self, context: PredictionContext) -> PredictionResult:
        if not isinstance(context, TrafficHistoryContext):
            raise TypeError("Bridge predictor requires TrafficHistoryContext")
        context.validate()
        current = np.asarray(context.current_dispatch_rows, dtype=np.int32)
        np.fill_diagonal(current, 0)
        auxiliary: dict[str, object] = {"bridge_mode": self.config.mode, "online_training": False}
        confidence = float(self.config.confidence)
        if self._artifact is None:
            predicted = current.copy()
        else:
            artifact = self._artifact
            if int(context.world_size) != artifact.world_size:
                raise ValueError("bridge artifact world size mismatch")
            source_layer = _source_layer(context.identity)
            if source_layer is None:
                raise ValueError("frozen bridge requires numeric source_layer_id")
            if source_layer + 1 >= artifact.max_layer:
                predicted = np.zeros_like(current)
                confidence = 0.0
                auxiliary["terminal_target_layer"] = True
            else:
                coef = np.asarray(artifact.coefficients, dtype=np.float64)
                intercept = np.asarray(artifact.intercept, dtype=np.float64)
                raw = (coef @ _feature(current, source_layer=source_layer, max_layer=artifact.max_layer) + intercept).reshape(current.shape)
                predicted = _round_remote(raw * max(float(current.sum()), 1.0), int(current.sum()))
                confidence = float(artifact.confidence)
            auxiliary.update(
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_digest": artifact.digest(),
                    "training_corpus_digest": artifact.training_corpus_digest,
                    "model_name": artifact.model_name,
                }
            )
        hint = PredictionHint(
            predictor_id=self.predictor_id,
            hint_type="copy_current" if self._artifact is None else "learned_prediction",
            target_dispatch_rows=_rows(predicted),
            confidence=confidence,
            oracle=False,
            source_layer_id=context.identity.source_layer_id,
            target_layer_id=context.identity.target_layer_id,
        )
        result = PredictionResult(identity=context.identity, hint=hint, expert_route=None, auxiliary=auxiliary)
        result.validate(world_size=int(context.world_size))
        return result


BRIDGE_COPY_SPEC = PredictorSpec(
    predictor_id="bridge_copy_current",
    category="traffic_matrix",
    deployable=True,
    offline_only=False,
    test_only=False,
    historical_aliases=("bridge", "future_bridge", "bridge_current"),
)
BRIDGE_AFFINE_SPEC = PredictorSpec(
    predictor_id="bridge_frozen_affine",
    category="traffic_matrix",
    deployable=True,
    offline_only=False,
    test_only=False,
    historical_aliases=("future_bridge_ridge_v1", "bridge_ridge"),
)


def bridge_predictor_specs() -> tuple[PredictorSpec, ...]:
    return (BRIDGE_COPY_SPEC, BRIDGE_AFFINE_SPEC)


def create_bridge_predictor(predictor_id: str, config: Mapping[str, object] | None = None) -> BridgeFormalPredictor:
    values = dict(config or {})
    normalized = str(predictor_id)
    affine_names = {BRIDGE_AFFINE_SPEC.predictor_id, *BRIDGE_AFFINE_SPEC.historical_aliases}
    values.setdefault("mode", "frozen_affine" if normalized in affine_names else "copy_current")
    return BridgeFormalPredictor(BridgeFormalConfig(**values))


__all__ = [
    "BRIDGE_AFFINE_SPEC",
    "BRIDGE_COPY_SPEC",
    "BridgeArtifact",
    "BridgeFormalConfig",
    "BridgeFormalPredictor",
    "bridge_predictor_specs",
    "create_bridge_predictor",
]
