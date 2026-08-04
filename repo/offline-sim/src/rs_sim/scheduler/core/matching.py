from __future__ import annotations

from typing import Callable


def _exact_dfs_matching(
    *,
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
    edge_weight: Callable[[int, int], float],
) -> tuple[tuple[int, int], ...]:
    best_weight = float("-inf")
    best_edges: tuple[tuple[int, int], ...] = ()

    def dfs(index: int, used_dst: set[int], chosen: list[tuple[int, int]], weight: float) -> None:
        nonlocal best_weight, best_edges
        if index >= len(sources):
            candidate = tuple(sorted(chosen))
            if weight > best_weight or (weight == best_weight and candidate < best_edges):
                best_weight = weight
                best_edges = candidate
            return
        src = sources[index]
        dfs(index + 1, used_dst, chosen, weight)
        for dst in destinations:
            if dst in used_dst:
                continue
            current = float(edge_weight(src, dst))
            if current <= 0.0:
                continue
            used_dst.add(dst)
            chosen.append((src, dst))
            dfs(index + 1, used_dst, chosen, weight + current)
            chosen.pop()
            used_dst.remove(dst)

    dfs(0, set(), [], 0.0)
    return best_edges


def _polynomial_matching(
    *,
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
    edge_weight: Callable[[int, int], float],
) -> tuple[tuple[int, int], ...]:
    """Deterministic O(n^3) maximum-weight assignment.

    A square matrix is padded with zero-weight edges; zero assignments are
    discarded afterwards, which represents unmatched endpoints. Floating
    scores are converted to fixed-point integers so the Hungarian loop has no
    tolerance-dependent behavior.
    """

    size = max(len(sources), len(destinations))
    if size == 0:
        return ()
    scores = [[0 for _ in range(size)] for _ in range(size)]
    max_score = 0
    for i, src in enumerate(sources):
        for j, dst in enumerate(destinations):
            value = max(0.0, float(edge_weight(src, dst)))
            score = int(round(value * 1_000_000_000))
            scores[i][j] = score
            max_score = max(max_score, score)
    costs = [[max_score - score for score in row] for row in scores]

    # Classic 1-indexed Hungarian algorithm for minimum-cost assignment.
    inf = 10**60
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 == 0:
                raise RuntimeError("Hungarian assignment made no progress")
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assigned_column_by_row = [0] * (size + 1)
    for j in range(1, size + 1):
        assigned_column_by_row[p[j]] = j
    chosen: list[tuple[int, int]] = []
    for i, src in enumerate(sources, 1):
        j = assigned_column_by_row[i]
        if 1 <= j <= len(destinations) and scores[i - 1][j - 1] > 0:
            chosen.append((src, destinations[j - 1]))
    return tuple(sorted(chosen))


def maximum_weight_bipartite_matching(
    *,
    sources: tuple[int, ...],
    destinations: tuple[int, ...],
    edge_weight: Callable[[int, int], float],
) -> tuple[tuple[int, int], ...]:
    """Deterministic maximum-weight matching for EP2 through EP32 and beyond.

    EP4 and smaller retain the original exact DFS path for reference
    compatibility. Larger EP sizes use an exact polynomial residual-network
    solver, avoiding the former exponential EP4-only limit.
    """

    srcs = tuple(int(src) for src in sources)
    dsts = tuple(int(dst) for dst in destinations)
    if len(srcs) <= 4 and len(dsts) <= 4:
        return _exact_dfs_matching(sources=srcs, destinations=dsts, edge_weight=edge_weight)
    return _polynomial_matching(sources=srcs, destinations=dsts, edge_weight=edge_weight)
