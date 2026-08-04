from __future__ import annotations

from typing import Callable

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


_EXACT_DFS_MAX_PORTS = 6


def maximum_weight_bipartite_matching(
    *,
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
    edge_weight: Callable[[int, int], float],
) -> tuple[tuple[int, int], ...]:
    srcs = tuple(int(src) for src in sources)
    dsts = tuple(int(dst) for dst in destinations)
    if linear_sum_assignment is not None and max(len(srcs), len(dsts)) > _EXACT_DFS_MAX_PORTS:
        return _scipy_maximum_weight_matching(srcs, dsts, edge_weight)
    return _dfs_maximum_weight_matching(srcs, dsts, edge_weight)


def _dfs_maximum_weight_matching(
    srcs: tuple[int, ...],
    dsts: tuple[int, ...],
    edge_weight: Callable[[int, int], float],
) -> tuple[tuple[int, int], ...]:
    best_weight = float("-inf")
    best_edges: tuple[tuple[int, int], ...] = ()

    def dfs(index: int, used_dst: set[int], chosen: list[tuple[int, int]], weight: float) -> None:
        nonlocal best_weight, best_edges
        if index >= len(srcs):
            candidate = tuple(chosen)
            candidate_sorted = tuple(sorted(candidate))
            if weight > best_weight or (weight == best_weight and candidate_sorted < tuple(sorted(best_edges))):
                best_weight = weight
                best_edges = candidate_sorted
            return
        src = srcs[index]
        dfs(index + 1, used_dst, chosen, weight)
        for dst in dsts:
            if dst in used_dst:
                continue
            w = float(edge_weight(src, dst))
            if w <= 0:
                continue
            used_dst.add(dst)
            chosen.append((src, dst))
            dfs(index + 1, used_dst, chosen, weight + w)
            chosen.pop()
            used_dst.remove(dst)

    dfs(0, set(), [], 0.0)
    return best_edges


def _scipy_maximum_weight_matching(
    srcs: tuple[int, ...],
    dsts: tuple[int, ...],
    edge_weight: Callable[[int, int], float],
) -> tuple[tuple[int, int], ...]:
    # Use scipy for larger port counts; the DFS implementation is exact but
    # exponential and blocks n=8 trace studies. A tiny deterministic penalty
    # gives stable tie-breaking without changing integer-weight optima.
    import numpy as np

    if not srcs or not dsts:
        return ()
    weights = np.zeros((len(srcs), len(dsts)), dtype=float)
    max_abs = 0.0
    for i, src in enumerate(srcs):
        for j, dst in enumerate(dsts):
            w = float(edge_weight(src, dst))
            if w > 0.0:
                weights[i, j] = w
                max_abs = max(max_abs, abs(w))
    if not np.any(weights > 0.0):
        return ()
    epsilon = max(max_abs, 1.0) * 1e-12
    tie_penalty = np.fromfunction(lambda i, j: (i * len(dsts) + j) * epsilon, weights.shape, dtype=float)
    cost = -(weights - tie_penalty)
    row_idx, col_idx = linear_sum_assignment(cost)
    edges = []
    for row, col in zip(row_idx, col_idx, strict=False):
        if weights[row, col] > 0.0:
            edges.append((srcs[int(row)], dsts[int(col)]))
    return tuple(sorted(edges))
