"""Canonical remote-only traffic-matrix helpers.

All communication scheduling paths should treat diagonal/self traffic as local
metadata rather than network traffic. This module centralizes that invariant.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable


MatrixLike = Iterable[Iterable[int]]
CanonicalMatrix = tuple[tuple[int, ...], ...]


def canonicalize_remote_matrix(matrix: MatrixLike, *, keep_shape: bool = True) -> CanonicalMatrix:
    rows = [tuple(max(0, int(value)) for value in row) for row in matrix]
    if not keep_shape:
        rows = [row for row in rows if row]
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    canonical: list[tuple[int, ...]] = []
    for src in range(height):
        padded = list(rows[src]) + [0] * max(0, width - len(rows[src]))
        for dst in range(width):
            if src == dst:
                padded[dst] = 0
        canonical.append(tuple(int(value) for value in padded))
    return tuple(canonical)


def matrix_total_bytes(matrix: MatrixLike) -> int:
    return int(sum(max(0, int(value)) for row in matrix for value in row))


def matrix_self_bytes(matrix: MatrixLike) -> int:
    rows = [tuple(max(0, int(value)) for value in row) for row in matrix]
    return int(sum(rows[idx][idx] for idx in range(min(len(rows), max((len(row) for row in rows), default=0))) if idx < len(rows[idx])))


def matrix_remote_bytes(matrix: MatrixLike) -> int:
    canonical = canonicalize_remote_matrix(matrix)
    return int(sum(sum(row) for row in canonical))


def matrix_nonzero_remote_edge_count(matrix: MatrixLike) -> int:
    canonical = canonicalize_remote_matrix(matrix)
    return int(sum(1 for row in canonical for value in row if int(value) > 0))


def matrix_row_sums_remote(matrix: MatrixLike) -> tuple[int, ...]:
    canonical = canonicalize_remote_matrix(matrix)
    return tuple(int(sum(row)) for row in canonical)


def matrix_col_sums_remote(matrix: MatrixLike) -> tuple[int, ...]:
    canonical = canonicalize_remote_matrix(matrix)
    if not canonical:
        return ()
    width = len(canonical[0])
    return tuple(int(sum(canonical[src][dst] for src in range(len(canonical)))) for dst in range(width))


def matrix_digest_remote(matrix: MatrixLike) -> str:
    canonical = canonicalize_remote_matrix(matrix)
    payload = json.dumps(canonical, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def matrix_diagonal_report(matrix: MatrixLike) -> dict[str, object]:
    rows = [tuple(max(0, int(value)) for value in row) for row in matrix]
    height = len(rows)
    width = max((len(row) for row in rows), default=0)
    self_bytes = matrix_self_bytes(rows)
    total_bytes = matrix_total_bytes(rows)
    remote_bytes = matrix_remote_bytes(rows)
    diagonal_nonzero_count = int(
        sum(1 for idx in range(min(height, width)) if idx < len(rows[idx]) and int(rows[idx][idx]) > 0)
    )
    return {
        "total_bytes": int(total_bytes),
        "remote_bytes": int(remote_bytes),
        "self_bytes": int(self_bytes),
        "self_byte_ratio": 0.0 if total_bytes <= 0 else float(self_bytes / total_bytes),
        "diagonal_nonzero_count": diagonal_nonzero_count,
        "shape": (height, width),
    }

