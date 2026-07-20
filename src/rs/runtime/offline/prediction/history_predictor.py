from __future__ import annotations

from dataclasses import dataclass

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix

from .contracts import Matrix, PredictorArtifact, PredictorSample


def _zeros_like(matrix: Matrix) -> Matrix:
    return canonicalize_remote_matrix(tuple(tuple(0 for _ in row) for row in matrix))


def _blend(left: Matrix, right: Matrix, alpha: float) -> Matrix:
    return canonicalize_remote_matrix(
        tuple(
            tuple(max(0, int(round(alpha * float(lv) + (1.0 - alpha) * float(rv)))) for lv, rv in zip(lrow, rrow, strict=False))
            for lrow, rrow in zip(left, right, strict=False)
        )
    )


@dataclass
class FATEStyleHistoryPredictor:
    alpha: float = 0.5
    predictor_name: str = "history_ema"
    predictor_version: str = "v1"
    _historical_mean: Matrix | None = None

    def fit(self, samples: list[PredictorSample]) -> "FATEStyleHistoryPredictor":
        if not samples:
            raise ValueError("history predictor requires at least one sample")
        accum = [[0.0 for _ in row] for row in samples[0].target_next_dispatch_matrix]
        for sample in samples:
            for src, row in enumerate(sample.target_next_dispatch_matrix):
                for dst, value in enumerate(row):
                    accum[src][dst] += float(value)
        count = float(len(samples))
        self._historical_mean = canonicalize_remote_matrix(tuple(tuple(int(round(value / count)) for value in row) for row in accum))
        return self

    def predict_matrix(self, sample: PredictorSample) -> Matrix:
        if self._historical_mean is None:
            return _zeros_like(sample.current_dispatch_matrix)
        return _blend(sample.current_dispatch_matrix, self._historical_mean, self.alpha)

    def to_artifact(self) -> PredictorArtifact:
        if self._historical_mean is None:
            raise ValueError("history predictor must be fit before export")
        return PredictorArtifact(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            feature_spec="ewma_current_dispatch_plus_historical_mean",
            world_size=len(self._historical_mean),
            metadata={"alpha": float(self.alpha)},
            payload={"historical_mean": [list(row) for row in self._historical_mean]},
        )

    @classmethod
    def from_artifact(cls, artifact: PredictorArtifact) -> "FATEStyleHistoryPredictor":
        predictor = cls(alpha=float(artifact.metadata.get("alpha", 0.7)))
        predictor._historical_mean = canonicalize_remote_matrix(artifact.payload["historical_mean"])
        return predictor
