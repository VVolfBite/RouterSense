from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rs.core.contracts import PredictionContext, PredictionResult


class Predictor(Protocol):
    @property
    def predictor_id(self) -> str:
        ...

    def predict(self, context: PredictionContext) -> PredictionResult:
        ...


@dataclass(frozen=True)
class PredictorSpec:
    predictor_id: str
    category: str
    deployable: bool
    offline_only: bool
    historical_aliases: tuple[str, ...] = ()


__all__ = ["Predictor", "PredictorSpec"]
