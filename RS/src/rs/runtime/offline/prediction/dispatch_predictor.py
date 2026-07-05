from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from rs.scheduling.contracts import ForecastPressure


class UnsupportedP2Predictor(RuntimeError):
    pass


def _normalize_matrix(matrix: list[list[int]] | tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _matrix_digest(matrix: tuple[tuple[int, ...], ...]) -> str:
    return hashlib.sha256(json.dumps(matrix).encode("utf-8")).hexdigest()[:16]


def _scale_matrix(matrix: tuple[tuple[int, ...], ...], *, scale: float) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(round(float(value) * float(scale))) for value in row) for row in matrix)


def _shuffle_matrix(matrix: tuple[tuple[int, ...], ...], *, seed: int = 42) -> tuple[tuple[int, ...], ...]:
    flat = [value for row in matrix for value in row]
    rng = random.Random(int(seed))
    rng.shuffle(flat)
    width = len(matrix[0]) if matrix else 0
    rows = [tuple(int(value) for value in flat[index:index + width]) for index in range(0, len(flat), width)]
    return tuple(rows)


def build_dispatch_forecast(
    *,
    mode: str,
    current_dispatch_matrix: list[list[int]] | tuple[tuple[int, ...], ...],
    actual_next_dispatch_matrix: list[list[int]] | tuple[tuple[int, ...], ...],
    scale: float = 1.0,
) -> ForecastPressure:
    current = _normalize_matrix(current_dispatch_matrix)
    actual_next = _normalize_matrix(actual_next_dispatch_matrix)
    if mode == "copy_current_dispatch":
        matrix = _scale_matrix(current, scale=scale)
        oracle = False
        evaluation_eligible = True
    elif mode == "perfect_trace":
        matrix = actual_next
        oracle = True
        evaluation_eligible = False
    elif mode == "zero_hint":
        matrix = tuple(tuple(0 for _ in row) for row in current)
        oracle = False
        evaluation_eligible = True
    elif mode == "shuffled_hint":
        matrix = _shuffle_matrix(actual_next)
        oracle = False
        evaluation_eligible = False
    elif mode == "calibrated_artifact":
        raise UnsupportedP2Predictor("calibrated_artifact is not implemented in the frozen scheduling core")
    else:
        raise ValueError(f"unsupported p2 forecast mode {mode!r}")
    total = sum(sum(int(value) for value in row) for row in matrix)
    shape = (len(matrix), len(matrix[0]) if matrix else 0)
    return ForecastPressure(
        source=mode,
        digest=_matrix_digest(matrix),
        oracle=oracle,
        evaluation_eligible=evaluation_eligible,
        matrix_shape=shape,
        matrix_total_bytes=int(total),
        matrix=matrix,
        metadata={"scale": float(scale)},
    )
