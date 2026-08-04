from __future__ import annotations

"""Deterministic integer metrics for P2 traffic prediction quality.

The simulator's authoritative evidence forbids floating-point values.  All
normalized scores are therefore reported in parts-per-million (ppm).
"""

from dataclasses import dataclass
from typing import Iterable

from rs_sim.scheduler.stable import stable_digest

_PPM = 1_000_000


def _normalize_matrix(matrix: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in matrix)
    if not rows:
        raise ValueError("prediction matrix must be non-empty")
    width = len(rows[0])
    if width <= 0 or any(len(row) != width for row in rows):
        raise ValueError("prediction matrix must be rectangular and non-empty")
    if any(value < 0 for row in rows for value in row):
        raise ValueError("prediction matrix values must be non-negative")
    return rows


def _ratio_ppm(numerator: int, denominator: int, *, zero_denominator_value: int) -> int:
    if denominator == 0:
        return int(zero_denominator_value)
    return (int(numerator) * _PPM) // int(denominator)


@dataclass(frozen=True, slots=True)
class P2PredictionQuality:
    predicted_total_bytes: int
    actual_total_bytes: int
    absolute_error_bytes: int
    relative_absolute_error_ppm: int
    matrix_overlap_ppm: int
    exact_edge_accuracy_ppm: int
    top_destination_accuracy_ppm: int
    nonzero_edge_precision_ppm: int
    nonzero_edge_recall_ppm: int
    quality_digest: str


def evaluate_p2_prediction(
    *,
    predicted_matrix: Iterable[Iterable[int]],
    actual_matrix: Iterable[Iterable[int]],
) -> P2PredictionQuality:
    predicted = _normalize_matrix(predicted_matrix)
    actual = _normalize_matrix(actual_matrix)
    if len(predicted) != len(actual) or len(predicted[0]) != len(actual[0]):
        raise ValueError("predicted and actual matrices must have identical shape")

    pairs = tuple(
        (predicted[src][dst], actual[src][dst])
        for src in range(len(actual))
        for dst in range(len(actual[0]))
    )
    predicted_total = sum(item[0] for item in pairs)
    actual_total = sum(item[1] for item in pairs)
    absolute_error = sum(abs(item[0] - item[1]) for item in pairs)
    overlap = sum(min(item[0], item[1]) for item in pairs)
    union = sum(max(item[0], item[1]) for item in pairs)
    exact_edges = sum(1 for item in pairs if item[0] == item[1])

    top_matches = 0
    for predicted_row, actual_row in zip(predicted, actual):
        # Stable tie-break: the lowest destination rank wins equal maxima.
        predicted_top = max(range(len(predicted_row)), key=lambda dst: (predicted_row[dst], -dst))
        actual_top = max(range(len(actual_row)), key=lambda dst: (actual_row[dst], -dst))
        if predicted_top == actual_top:
            top_matches += 1

    predicted_nonzero = {index for index, item in enumerate(pairs) if item[0] > 0}
    actual_nonzero = {index for index, item in enumerate(pairs) if item[1] > 0}
    true_positive = len(predicted_nonzero & actual_nonzero)

    payload = {
        "schema_version": "P2_PREDICTION_QUALITY",
        "predicted_total_bytes": predicted_total,
        "actual_total_bytes": actual_total,
        "absolute_error_bytes": absolute_error,
        "relative_absolute_error_ppm": _ratio_ppm(
            absolute_error,
            actual_total,
            zero_denominator_value=0 if predicted_total == 0 else _PPM,
        ),
        "matrix_overlap_ppm": _ratio_ppm(
            overlap,
            union,
            zero_denominator_value=_PPM,
        ),
        "exact_edge_accuracy_ppm": _ratio_ppm(exact_edges, len(pairs), zero_denominator_value=_PPM),
        "top_destination_accuracy_ppm": _ratio_ppm(
            top_matches,
            len(actual),
            zero_denominator_value=_PPM,
        ),
        "nonzero_edge_precision_ppm": _ratio_ppm(
            true_positive,
            len(predicted_nonzero),
            zero_denominator_value=_PPM if not actual_nonzero else 0,
        ),
        "nonzero_edge_recall_ppm": _ratio_ppm(
            true_positive,
            len(actual_nonzero),
            zero_denominator_value=_PPM,
        ),
    }
    fields = {key: value for key, value in payload.items() if key != "schema_version"}
    return P2PredictionQuality(
        **fields,
        quality_digest=stable_digest(payload),
    )


__all__ = ["P2PredictionQuality", "evaluate_p2_prediction"]
