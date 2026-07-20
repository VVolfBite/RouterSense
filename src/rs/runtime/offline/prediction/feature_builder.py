from __future__ import annotations

from pathlib import Path
from typing import Any

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_col_sums_remote, matrix_row_sums_remote

from .contracts import Matrix, PredictorSample


def normalize_matrix(matrix: Any) -> Matrix:
    return canonicalize_remote_matrix(matrix)


def row_sums(matrix: Matrix) -> tuple[int, ...]:
    return matrix_row_sums_remote(matrix)


def col_sums(matrix: Matrix) -> tuple[int, ...]:
    return matrix_col_sums_remote(matrix)


def flatten_matrix(matrix: Matrix) -> list[float]:
    return [float(value) for row in matrix for value in row]


def build_feature_vector(sample: PredictorSample) -> list[float]:
    current = flatten_matrix(sample.current_dispatch_matrix)
    returns = flatten_matrix(sample.current_return_matrix)
    previous = flatten_matrix(sample.previous_dispatch_matrix)
    return [
        *current,
        *returns,
        *previous,
        *[float(value) for value in row_sums(sample.current_dispatch_matrix)],
        *[float(value) for value in col_sums(sample.current_dispatch_matrix)],
        *[float(value) for value in row_sums(sample.current_return_matrix)],
        *[float(value) for value in col_sums(sample.current_return_matrix)],
        float(sum(current)),
        float(sum(returns)),
        float(int(sample.layer_id) if str(sample.layer_id).isdigit() else 0),
    ]


def load_fixture_samples(fixture_dir: Path) -> list[PredictorSample]:
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    raw = [__import__("json").loads(path.read_text(encoding="utf-8")) for path in fixture_paths]
    samples: list[PredictorSample] = []
    prev_dispatch: Matrix | None = None
    for fixture in raw:
        current = normalize_matrix(fixture["p0_dispatch_matrix"])
        returns = normalize_matrix(fixture["p1_return_matrix"])
        target = normalize_matrix(fixture.get("p2_next_dispatch_matrix", fixture.get("p2_next_dispatch_forecast_matrix", [])))
        if prev_dispatch is None:
            prev_dispatch = tuple(tuple(0 for _ in row) for row in current)
        meta = fixture.get("metadata", {})
        samples.append(
            PredictorSample(
                layer_id=str(meta.get("layer_id", "")),
                next_layer_id=str(meta.get("next_layer_id", "")),
                current_dispatch_matrix=current,
                current_return_matrix=returns,
                previous_dispatch_matrix=prev_dispatch,
                target_next_dispatch_matrix=target,
            )
        )
        prev_dispatch = current
    return samples
