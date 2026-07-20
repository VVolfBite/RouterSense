"""Construction helpers for uncertainty-aware traffic forecasts."""

from __future__ import annotations

from collections.abc import Sequence
import math

from rs.core.contracts import (
    MatrixRows,
    PredictionHint,
    RankPressureForecast,
    StableEdgePrecedence,
    TrafficForecastEnvelope,
)


def _rows(matrix: Sequence[Sequence[int]]) -> MatrixRows:
    rows = tuple(tuple(int(value) for value in row) for row in matrix)
    size = len(rows)
    if size <= 0 or any(len(row) != size for row in rows):
        raise ValueError("traffic matrix must be non-empty and square")
    if any(value < 0 for row in rows for value in row):
        raise ValueError("traffic matrix must be non-negative")
    return rows


def _remote_row_sum(matrix: MatrixRows, source: int) -> float:
    return float(sum(value for destination, value in enumerate(matrix[source]) if destination != source))


def build_traffic_forecast_envelope(
    *,
    predictor_id: str,
    mean_rows: Sequence[Sequence[int]],
    lower_rows: Sequence[Sequence[int]] | None = None,
    upper_rows: Sequence[Sequence[int]] | None = None,
    relative_error_bound: float | Sequence[float] | None = None,
    precedence_margin: float = 0.0,
    precedence_confidence_floor: float = 0.5,
    calibration_id: str = "",
    source_layer_id: str | None = None,
    target_layer_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> TrafficForecastEnvelope:
    """Build a calibrated rank-level forecast envelope.

    Callers may provide explicit lower/upper matrices or a per-source relative
    error bound. Bounds are integer count bounds and never create executable
    traffic. Stable precedence is emitted only when two edge intervals do not
    overlap by at least ``precedence_margin``.
    """

    mean = _rows(mean_rows)
    size = len(mean)
    if (lower_rows is None) != (upper_rows is None):
        raise ValueError("lower_rows and upper_rows must be provided together")
    if lower_rows is not None:
        lower = _rows(lower_rows)
        upper = _rows(upper_rows or ())
    else:
        if relative_error_bound is None:
            bounds = [0.0 for _ in range(size)]
        elif isinstance(relative_error_bound, Sequence) and not isinstance(relative_error_bound, (str, bytes)):
            bounds = [float(value) for value in relative_error_bound]
            if len(bounds) != size:
                raise ValueError("relative_error_bound width must match world size")
        else:
            bounds = [float(relative_error_bound) for _ in range(size)]
        if any(not math.isfinite(value) or value < 0.0 for value in bounds):
            raise ValueError("relative error bounds must be finite and non-negative")
        lower_rows_mut: list[list[int]] = []
        upper_rows_mut: list[list[int]] = []
        for source, row in enumerate(mean):
            error = bounds[source]
            lower_rows_mut.append([max(0, int(math.floor(value * (1.0 - error)))) for value in row])
            upper_rows_mut.append([max(0, int(math.ceil(value * (1.0 + error)))) for value in row])
        lower = _rows(lower_rows_mut)
        upper = _rows(upper_rows_mut)

    pressure_mean = tuple(_remote_row_sum(mean, source) for source in range(size))
    pressure_lower = tuple(_remote_row_sum(lower, source) for source in range(size))
    pressure_upper = tuple(_remote_row_sum(upper, source) for source in range(size))
    pressure_confidence = tuple(
        max(0.0, min(1.0, 1.0 - (upper_value - lower_value) / max(upper_value, 1.0)))
        for lower_value, upper_value in zip(pressure_lower, pressure_upper, strict=True)
    )
    rank_pressure = RankPressureForecast(
        mean=pressure_mean,
        lower=pressure_lower,
        upper=pressure_upper,
        confidence=pressure_confidence,
    )

    edges = [
        (source, destination)
        for source in range(size)
        for destination in range(size)
        if source != destination and upper[source][destination] > 0
    ]
    precedence: list[StableEdgePrecedence] = []
    margin_floor = max(0.0, float(precedence_margin))
    confidence_floor = max(0.0, min(1.0, float(precedence_confidence_floor)))
    for before_src, before_dst in edges:
        for after_src, after_dst in edges:
            if (before_src, before_dst) == (after_src, after_dst):
                continue
            margin = float(lower[before_src][before_dst] - upper[after_src][after_dst])
            if margin <= margin_floor:
                continue
            width = float(
                (upper[before_src][before_dst] - lower[before_src][before_dst])
                + (upper[after_src][after_dst] - lower[after_src][after_dst])
            )
            confidence = max(0.0, min(1.0, margin / max(margin + width, 1.0)))
            if confidence < confidence_floor:
                continue
            precedence.append(
                StableEdgePrecedence(
                    before_src=before_src,
                    before_dst=before_dst,
                    after_src=after_src,
                    after_dst=after_dst,
                    margin=margin,
                    confidence=confidence,
                )
            )

    envelope = TrafficForecastEnvelope(
        predictor_id=str(predictor_id),
        mean_rows=mean,
        lower_rows=lower,
        upper_rows=upper,
        rank_pressure=rank_pressure,
        stable_precedence=tuple(precedence),
        calibration_id=str(calibration_id),
        source_layer_id=source_layer_id,
        target_layer_id=target_layer_id,
        metadata=dict(metadata or {}),
    )
    envelope.validate(world_size=size)
    return envelope


def evaluate_traffic_forecast(
    predicted_rows: Sequence[Sequence[int]],
    actual_rows: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Return model-agnostic traffic and rank-pressure forecast metrics."""

    predicted = _rows(predicted_rows)
    actual = _rows(actual_rows)
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual traffic shapes must match")
    size = len(actual)
    all_error = 0.0
    all_actual = 0.0
    remote_error = 0.0
    remote_actual = 0.0
    predicted_pressure = []
    actual_pressure = []
    for source in range(size):
        if len(predicted[source]) != len(actual[source]):
            raise ValueError("predicted and actual traffic shapes must match")
        pred_remote = 0.0
        act_remote = 0.0
        for destination in range(size):
            pred = float(predicted[source][destination])
            act = float(actual[source][destination])
            all_error += abs(pred - act)
            all_actual += act
            if source != destination:
                remote_error += abs(pred - act)
                remote_actual += act
                pred_remote += pred
                act_remote += act
        predicted_pressure.append(pred_remote)
        actual_pressure.append(act_remote)
    pressure_error = sum(
        abs(pred - act)
        for pred, act in zip(predicted_pressure, actual_pressure, strict=True)
    )
    pressure_actual = sum(actual_pressure)
    pred_remote_total = sum(predicted_pressure)
    hot_pred = max(range(size), key=lambda rank: (predicted_pressure[rank], -rank))
    hot_actual = max(range(size), key=lambda rank: (actual_pressure[rank], -rank))
    dot = sum(a * b for a, b in zip(predicted_pressure, actual_pressure, strict=True))
    pred_norm = math.sqrt(sum(value * value for value in predicted_pressure))
    actual_norm = math.sqrt(sum(value * value for value in actual_pressure))
    return {
        "all_relative_l1": 0.0 if all_actual <= 0.0 else all_error / all_actual,
        "remote_relative_l1": 0.0 if remote_actual <= 0.0 else remote_error / remote_actual,
        "rank_pressure_relative_l1": (
            0.0 if pressure_actual <= 0.0 else pressure_error / pressure_actual
        ),
        "remote_total_bias": (
            0.0 if remote_actual <= 0.0 else (pred_remote_total - remote_actual) / remote_actual
        ),
        "rank_pressure_cosine": (
            0.0 if pred_norm <= 0.0 or actual_norm <= 0.0 else dot / (pred_norm * actual_norm)
        ),
        "hot_source_top1_match": 1.0 if hot_pred == hot_actual else 0.0,
    }


def envelope_to_prediction_hint(
    envelope: TrafficForecastEnvelope,
    *,
    confidence: float | None = None,
) -> PredictionHint:
    """Project an envelope to the legacy matrix hint without losing provenance."""

    envelope.validate()
    if confidence is None:
        values = envelope.rank_pressure.confidence
        confidence = sum(values) / len(values) if values else 0.0
    hint = PredictionHint(
        predictor_id=envelope.predictor_id,
        hint_type="learned_prediction",
        target_dispatch_rows=envelope.mean_rows,
        confidence=max(0.0, min(1.0, float(confidence))),
        oracle=False,
        source_layer_id=envelope.source_layer_id,
        target_layer_id=envelope.target_layer_id,
        prediction_kind="learned_prediction",
    )
    hint.validate(world_size=len(envelope.mean_rows))
    return hint


__all__ = ["build_traffic_forecast_envelope", "envelope_to_prediction_hint", "evaluate_traffic_forecast"]
