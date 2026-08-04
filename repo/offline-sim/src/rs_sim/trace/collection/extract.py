"""Framework-neutral routing-map extraction helpers.

The helpers accept Torch tensors, NumPy arrays, or nested Python sequences.
Only compact per-expert counts are copied to the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class RoutingExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class RoutingCountResult:
    raw_selected_rows: tuple[int, ...]
    kept_rows: tuple[int, ...]
    dropped_rows: tuple[int, ...]
    padding_rows: tuple[int, ...]
    num_experts: int
    source_shape: tuple[int, ...]
    extraction_mode: str


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(v) for v in shape)
    data = _to_list(value)
    dims: list[int] = []
    cursor = data
    while isinstance(cursor, (list, tuple)):
        dims.append(len(cursor))
        cursor = cursor[0] if cursor else []
    return tuple(dims)


def _to_list(value: Any) -> Any:
    current = value
    for method in ("detach", "cpu"):
        fn = getattr(current, method, None)
        if callable(fn):
            current = fn()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(current, (list, tuple)):
        return [(_to_list(item) if isinstance(item, (list, tuple)) else item) for item in current]
    raise RoutingExtractionError(f"unsupported tensor/array type: {type(value).__name__}")


def _bool_matrix(value: Any) -> list[list[bool]]:
    data = _to_list(value)
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise RoutingExtractionError("routing map must have rank >= 2")
    shape = _shape(value)
    if len(shape) == 2:
        return [[bool(cell) for cell in row] for row in data]
    if len(shape) == 3:
        flattened: list[list[bool]] = []
        for token in data:
            row: list[bool] = []
            for rank_entries in token:
                row.extend(bool(cell) for cell in rank_entries)
            flattened.append(row)
        return flattened
    raise RoutingExtractionError(f"routing map rank {len(shape)} is unsupported; expected 2D or 3D")


def _float_matrix(value: Any, *, expected_columns: int) -> list[list[float]] | None:
    if value is None:
        return None
    data = _to_list(value)
    shape = _shape(value)
    if len(shape) == 2:
        matrix = [[float(cell) for cell in row] for row in data]
    elif len(shape) == 3:
        matrix = []
        for token in data:
            row: list[float] = []
            for rank_entries in token:
                row.extend(float(cell) for cell in rank_entries)
            matrix.append(row)
    else:
        return None
    if matrix and len(matrix[0]) != expected_columns:
        return None
    return matrix


def _column_counts(matrix: list[list[bool]]) -> list[int]:
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise RoutingExtractionError("routing map is ragged")
    return [sum(1 for row in matrix if row[column]) for column in range(columns)]


def _normalize_vector(value: Iterable[int] | None, *, length: int, name: str) -> list[int] | None:
    if value is None:
        return None
    result = [int(v) for v in value]
    if len(result) != length or any(v < 0 for v in result):
        raise RoutingExtractionError(f"{name} must be a nonnegative vector of length {length}")
    return result


def extract_routing_counts(
    *,
    routing_map: Any,
    probs: Any | None = None,
    raw_routing_map: Any | None = None,
    explicit_padding_rows: Iterable[int] | None = None,
    drop_and_pad: bool = False,
    infer_padding_from_zero_prob: bool = False,
) -> RoutingCountResult:
    final_matrix = _bool_matrix(routing_map)
    final_counts = _column_counts(final_matrix)
    columns = len(final_counts)
    padding = _normalize_vector(explicit_padding_rows, length=columns, name="explicit_padding_rows")

    if padding is None and drop_and_pad and infer_padding_from_zero_prob:
        probability_matrix = _float_matrix(probs, expected_columns=columns)
        if probability_matrix is not None:
            padding = [
                sum(
                    1
                    for token in range(len(final_matrix))
                    if final_matrix[token][column] and abs(probability_matrix[token][column]) == 0.0
                )
                for column in range(columns)
            ]
    if padding is None:
        padding = [0] * columns

    kept = [final_counts[index] - padding[index] for index in range(columns)]
    if any(value < 0 for value in kept):
        raise RoutingExtractionError("padding rows exceed final routing-map counts")

    if raw_routing_map is not None:
        raw_matrix = _bool_matrix(raw_routing_map)
        raw = _column_counts(raw_matrix)
        if len(raw) != columns:
            raise RoutingExtractionError("raw and final routing maps have different expert dimensions")
        dropped = [raw[index] - kept[index] for index in range(columns)]
        if any(value < 0 for value in dropped):
            raise RoutingExtractionError("raw routing counts are smaller than kept counts")
        mode = "explicit_raw_final_and_padding"
    else:
        raw = list(kept)
        dropped = [0] * columns
        mode = "exact_realized_final_map_raw_drop_unavailable"

    return RoutingCountResult(
        raw_selected_rows=tuple(raw),
        kept_rows=tuple(kept),
        dropped_rows=tuple(dropped),
        padding_rows=tuple(padding),
        num_experts=columns,
        source_shape=_shape(routing_map),
        extraction_mode=mode,
    )
