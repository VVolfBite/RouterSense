"""Matching helpers for multiphase scheduling."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def build_weight_matrix(
    ready_flows: list[dict[str, Any]],
    num_gpus: int,
) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, Any]]]:
    weights = np.zeros((num_gpus, num_gpus), dtype=float)
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in ready_flows:
        src = int(candidate["src_gpu"])
        dst = int(candidate["dst_gpu"])
        score = float(candidate["score"])
        key = (src, dst)
        if score > weights[src, dst]:
            weights[src, dst] = score
            selected[key] = candidate
    return weights, selected


def greedy_maximal_matching(ready_flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used_src: set[int] = set()
    used_dst: set[int] = set()
    for candidate in sorted(
        ready_flows,
        key=lambda item: (
            float(item["score"]),
            float(item["barrier_urgency"]),
            float(item["age"]),
            float(item["residual"]),
            -int(item["src_gpu"]),
            -int(item["dst_gpu"]),
        ),
        reverse=True,
    ):
        src = int(candidate["src_gpu"])
        dst = int(candidate["dst_gpu"])
        if src in used_src or dst in used_dst:
            continue
        used_src.add(src)
        used_dst.add(dst)
        chosen.append(candidate)
    return chosen


def maximum_weight_matching(
    ready_flows: list[dict[str, Any]],
    num_gpus: int,
) -> list[dict[str, Any]]:
    if not ready_flows:
        return []
    weights, selected = build_weight_matrix(ready_flows, num_gpus)
    if linear_sum_assignment is None:  # pragma: no cover
        return greedy_maximal_matching(ready_flows)
    cost = -weights
    row_idx, col_idx = linear_sum_assignment(cost)
    result: list[dict[str, Any]] = []
    for src, dst in zip(row_idx, col_idx, strict=False):
        key = (int(src), int(dst))
        candidate = selected.get(key)
        if candidate is None or int(src) == int(dst) or weights[src, dst] <= 0.0:
            continue
        result.append(candidate)
    return result
