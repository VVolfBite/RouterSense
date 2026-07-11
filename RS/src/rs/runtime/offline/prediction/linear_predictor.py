from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix

from .contracts import Matrix, PredictorArtifact, PredictorSample
from .feature_builder import build_feature_vector, flatten_matrix


def _matrix_shape(sample: PredictorSample) -> tuple[int, int]:
    return (len(sample.target_next_dispatch_matrix), len(sample.target_next_dispatch_matrix[0]) if sample.target_next_dispatch_matrix else 0)


def _reshape(values: list[float], *, rows: int, cols: int) -> Matrix:
    clipped = [max(0, int(round(value))) for value in values]
    return canonicalize_remote_matrix(tuple(tuple(clipped[row * cols + col] for col in range(cols)) for row in range(rows)))


@dataclass
class FATEStyleLinearTrafficPredictor:
    ridge_lambda: float = 1e-3
    predictor_name: str = "ridge_linear_trace_predictor"
    predictor_version: str = "v1"
    _weight: torch.Tensor | None = None
    _bias: torch.Tensor | None = None
    _shape: tuple[int, int] | None = None

    def fit(self, samples: list[PredictorSample]) -> "FATEStyleLinearTrafficPredictor":
        if not samples:
            raise ValueError("linear predictor requires at least one sample")
        features = torch.tensor([build_feature_vector(sample) for sample in samples], dtype=torch.float64)
        targets = torch.tensor([flatten_matrix(sample.target_next_dispatch_matrix) for sample in samples], dtype=torch.float64)
        ones = torch.ones((features.shape[0], 1), dtype=torch.float64)
        design = torch.cat([features, ones], dim=1)
        eye = torch.eye(design.shape[1], dtype=torch.float64)
        eye[-1, -1] = 0.0
        normal_matrix = design.T @ design + self.ridge_lambda * eye
        rhs = design.T @ targets
        try:
            solution = torch.linalg.solve(normal_matrix, rhs)
        except RuntimeError:
            solution = torch.linalg.pinv(normal_matrix) @ rhs
        self._weight = solution[:-1, :]
        self._bias = solution[-1, :]
        self._shape = _matrix_shape(samples[0])
        return self

    def predict_matrix(self, sample: PredictorSample) -> Matrix:
        if self._weight is None or self._bias is None or self._shape is None:
            raise ValueError("linear predictor must be fit before predict")
        feature = torch.tensor(build_feature_vector(sample), dtype=torch.float64)
        output = feature @ self._weight + self._bias
        rows, cols = self._shape
        return _reshape(output.tolist(), rows=rows, cols=cols)

    def to_artifact(self) -> PredictorArtifact:
        if self._weight is None or self._bias is None or self._shape is None:
            raise ValueError("linear predictor must be fit before export")
        return PredictorArtifact(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            feature_spec="dispatch_return_prev_dispatch_rowcol_totals_layerid",
            world_size=int(self._shape[0]),
            metadata={"ridge_lambda": float(self.ridge_lambda), "shape": list(self._shape)},
            payload={"weight": self._weight.tolist(), "bias": self._bias.tolist()},
        )

    @classmethod
    def from_artifact(cls, artifact: PredictorArtifact) -> "FATEStyleLinearTrafficPredictor":
        predictor = cls(ridge_lambda=float(artifact.metadata.get("ridge_lambda", 1e-3)))
        predictor._shape = tuple(int(v) for v in artifact.metadata.get("shape", (0, 0)))
        predictor._weight = torch.tensor(artifact.payload["weight"], dtype=torch.float64)
        predictor._bias = torch.tensor(artifact.payload["bias"], dtype=torch.float64)
        return predictor
