#!/usr/bin/env python3
"""Shared helpers for timeline/prediction diagnosis scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


def pct_delta(baseline: float | int | None, value: float | int | None) -> float | None:
    if baseline in (None, 0) or value is None:
        return None
    return float((float(value) - float(baseline)) / float(baseline))


def matrix_row_sums(matrix: list[list[int]] | tuple[tuple[int, ...], ...]) -> list[int]:
    return [int(sum(int(v) for v in row)) for row in matrix]


def matrix_col_sums(matrix: list[list[int]] | tuple[tuple[int, ...], ...]) -> list[int]:
    size = len(matrix)
    return [int(sum(int(matrix[src][dst]) for src in range(size))) for dst in range(size)]


def top_edges(matrix: list[list[int]] | tuple[tuple[int, ...], ...], *, topk: int = 4) -> list[dict[str, int]]:
    edges: list[tuple[int, int, int]] = []
    for src, row in enumerate(matrix):
        for dst, value in enumerate(row):
            value_i = int(value)
            if src == dst or value_i <= 0:
                continue
            edges.append((src, dst, value_i))
    edges.sort(key=lambda item: (-item[2], item[0], item[1]))
    return [
        {"src_rank": src, "dst_rank": dst, "byte_count": value}
        for src, dst, value in edges[:topk]
    ]


def flatten_topology_signature(matrix: list[list[int]] | tuple[tuple[int, ...], ...]) -> dict[str, Any]:
    row_sums = matrix_row_sums(matrix)
    col_sums = matrix_col_sums(matrix)
    bottleneck_src = max(range(len(row_sums)), key=lambda idx: row_sums[idx]) if row_sums else None
    bottleneck_dst = max(range(len(col_sums)), key=lambda idx: col_sums[idx]) if col_sums else None
    return {
        "row_sums": row_sums,
        "col_sums": col_sums,
        "top_edges": top_edges(matrix),
        "bottleneck_src_rank": bottleneck_src,
        "bottleneck_dst_rank": bottleneck_dst,
    }

