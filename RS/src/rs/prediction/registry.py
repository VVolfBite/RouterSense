from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .api import Predictor, PredictorSpec, TrafficPredictionTrainingSample
from .expert_route import MockGateReplayExpertRoutePredictor
from .traffic_matrix import CopyCurrentTrafficPredictor, HistoryTrafficPredictor, LinearTrafficPredictor, ZeroTrafficPredictor


_SPECS = {
    "zero": PredictorSpec("zero", category="traffic_matrix", deployable=True, offline_only=False, historical_aliases=("zero_hint", "none")),
    "copy_current": PredictorSpec("copy_current", category="traffic_matrix", deployable=True, offline_only=False, historical_aliases=("copy_current_dispatch",)),
    "history": PredictorSpec("history", category="traffic_matrix", deployable=True, offline_only=False, historical_aliases=("history_ema", "fate_style_history")),
    "linear": PredictorSpec("linear", category="traffic_matrix", deployable=False, offline_only=True, historical_aliases=("ridge_linear_trace_predictor", "fate_style_linear", "history_linear_trend")),
    "mock_gate_replay": PredictorSpec("mock_gate_replay", category="expert_route", deployable=False, offline_only=True, test_only=True, historical_aliases=("MockGateReplayPredictor",)),
}

_ALIASES = {alias: spec_id for spec_id, spec in _SPECS.items() for alias in spec.historical_aliases}


def resolve_predictor_id(name: str) -> str:
    normalized = str(name).strip()
    if normalized in _SPECS:
        return normalized
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    raise ValueError(f"unknown predictor_id {name!r}")


class PredictionRegistry:
    @staticmethod
    def specs() -> tuple[PredictorSpec, ...]:
        return tuple(_SPECS.values())

    @staticmethod
    def create(predictor_id: str, config: Any | None = None, *, usage: str = "runtime") -> Predictor:
        resolved = resolve_predictor_id(predictor_id)
        spec = _SPECS[resolved]
        normalized_usage = str(usage)
        if normalized_usage not in {"runtime", "offline", "test"}:
            raise ValueError(f"unsupported predictor usage {usage!r}")
        if spec.test_only and normalized_usage != "test":
            raise ValueError(f"predictor {resolved!r} is test_only and may only be used with usage='test'")
        if spec.offline_only and normalized_usage == "runtime":
            raise ValueError(f"predictor {resolved!r} is offline_only and may not be used with usage='runtime'")
        if resolved == "zero":
            return ZeroTrafficPredictor()
        if resolved == "copy_current":
            return CopyCurrentTrafficPredictor()
        if resolved == "history":
            alpha = 0.5 if config is None else float(getattr(config, "alpha", config.get("alpha", 0.5)) if isinstance(config, dict) else getattr(config, "alpha", 0.5))
            return HistoryTrafficPredictor(alpha=alpha)
        if resolved == "linear":
            ridge_lambda = 1e-3 if config is None else float(getattr(config, "ridge_lambda", config.get("ridge_lambda", 1e-3)) if isinstance(config, dict) else getattr(config, "ridge_lambda", 1e-3))
            predictor = LinearTrafficPredictor(ridge_lambda=ridge_lambda)
            sample_rows = None
            if isinstance(config, dict):
                sample_rows = config.get("samples")
            else:
                sample_rows = getattr(config, "samples", None)
            if sample_rows:
                predictor.fit(tuple(_coerce_training_sample(item) for item in sample_rows))
            return predictor
        if resolved == "mock_gate_replay":
            return MockGateReplayExpertRoutePredictor()
        raise ValueError(f"unsupported predictor_id {predictor_id!r}")


__all__ = ["PredictionRegistry", "resolve_predictor_id"]


def _coerce_training_sample(value: object) -> TrafficPredictionTrainingSample:
    if isinstance(value, TrafficPredictionTrainingSample):
        return value
    sample = getattr(value, "__dict__", None)
    if isinstance(value, dict):
        sample = value
    if not isinstance(sample, dict):
        raise TypeError(f"unsupported training sample {type(value).__name__}")
    return TrafficPredictionTrainingSample(
        current_dispatch_rows=tuple(tuple(int(item) for item in row) for row in sample["current_dispatch_rows"]),
        current_return_rows=tuple(tuple(int(item) for item in row) for row in sample["current_return_rows"]),
        history_dispatch_rows=tuple(
            tuple(tuple(int(item) for item in row) for row in matrix)
            for matrix in sample.get("history_dispatch_rows", ())
        ),
        target_next_dispatch_rows=tuple(tuple(int(item) for item in row) for row in sample["target_next_dispatch_rows"]),
        layer_id=None if sample.get("layer_id") is None else str(sample.get("layer_id")),
        next_layer_id=None if sample.get("next_layer_id") is None else str(sample.get("next_layer_id")),
    )
