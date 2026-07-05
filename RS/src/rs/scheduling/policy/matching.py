from __future__ import annotations

from typing import Callable


def maximum_weight_bipartite_matching(
    *,
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
    edge_weight: Callable[[int, int], float],
) -> tuple[tuple[int, int], ...]:
    best_weight = float("-inf")
    best_edges: tuple[tuple[int, int], ...] = ()

    srcs = tuple(int(src) for src in sources)
    dsts = tuple(int(dst) for dst in destinations)

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
