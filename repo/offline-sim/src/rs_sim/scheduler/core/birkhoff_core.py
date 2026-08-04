from __future__ import annotations

"""Pure deterministic Δ-regular Birkhoff decomposition.

Source lineage:
``src/rs/scheduling/phase_local/birkhoff_phase_local.py``
``BirkhoffPhaseLocalPolicy._logical_decompose``.
"""

from dataclasses import dataclass
from typing import Iterable

from .digest import stable_digest
from .matching import maximum_weight_bipartite_matching


@dataclass(frozen=True, slots=True)
class BirkhoffInterval:
    interval_id: int
    coefficient: int
    real_edges: tuple[tuple[int, int], ...]
    dummy_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class BirkhoffCertificate:
    rank_count: int
    delta: int
    total_coefficient: int
    coverage_verified: bool
    matching_constraints_verified: bool
    intervals: tuple[BirkhoffInterval, ...]
    certificate_digest: str


@dataclass(frozen=True, slots=True)
class BirkhoffTask:
    task_id: str
    src_rank: int
    dst_rank: int
    payload_units: int
    chunk_index: int = 0
    byte_offset: int = 0


def _matrix(matrix: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in matrix)
    if not rows:
        return ()
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise ValueError("Birkhoff matrix must be square")
    if any(value < 0 for row in rows for value in row):
        raise ValueError("Birkhoff matrix values must be non-negative")
    return rows


def decompose_integer_matrix(matrix: Iterable[Iterable[int]]) -> BirkhoffCertificate:
    """Complete an integer traffic matrix to Δ-regular support and decompose it.

    Diagonal demand is local assembly and is excluded.  Every interval is a
    one-to-one matching; interval coefficients sum exactly to
    ``Delta=max(max row load,max column load)``.
    """

    normalized = _matrix(matrix)
    n = len(normalized)
    if n == 0:
        payload = {"rank_count": 0, "delta": 0, "intervals": ()}
        return BirkhoffCertificate(0, 0, 0, True, True, (), stable_digest(payload))
    real = [
        [0 if src == dst else int(normalized[src][dst]) for dst in range(n)]
        for src in range(n)
    ]
    original = tuple(tuple(row) for row in real)
    row_loads = [sum(row) for row in real]
    col_loads = [sum(real[src][dst] for src in range(n)) for dst in range(n)]
    delta = max(max(row_loads, default=0), max(col_loads, default=0))
    if delta <= 0:
        payload = {"rank_count": n, "delta": 0, "intervals": ()}
        return BirkhoffCertificate(n, 0, 0, True, True, (), stable_digest(payload))

    row_deficit = [delta - value for value in row_loads]
    col_deficit = [delta - value for value in col_loads]
    dummy = [[0 for _ in range(n)] for _ in range(n)]
    src = dst = 0
    while src < n and dst < n:
        if row_deficit[src] <= 0:
            src += 1
            continue
        if col_deficit[dst] <= 0:
            dst += 1
            continue
        amount = min(row_deficit[src], col_deficit[dst])
        dummy[src][dst] += amount
        row_deficit[src] -= amount
        col_deficit[dst] -= amount
    if any(row_deficit) or any(col_deficit):
        raise ValueError("Birkhoff regular completion failed")

    intervals: list[BirkhoffInterval] = []
    remaining_degree = delta
    ranks = tuple(range(n))
    total_coefficient = 0
    while remaining_degree > 0:
        cardinality_base = float(n + 2)

        def weight(src_rank: int, dst_rank: int) -> float:
            if real[src_rank][dst_rank] > 0:
                return cardinality_base + 1.0
            if dummy[src_rank][dst_rank] > 0:
                return cardinality_base
            return 0.0

        matching = maximum_weight_bipartite_matching(
            sources=ranks, destinations=ranks, edge_weight=weight
        )
        if len(matching) != n:
            raise ValueError("Birkhoff regular support did not yield a perfect matching")
        selected: list[tuple[int, int, str, int]] = []
        for source, destination in matching:
            if real[source][destination] > 0:
                selected.append((source, destination, "real", real[source][destination]))
            elif dummy[source][destination] > 0:
                selected.append((source, destination, "dummy", dummy[source][destination]))
            else:
                raise ValueError("Birkhoff matching selected zero-capacity edge")
        coefficient = min(capacity for _, _, _, capacity in selected)
        if coefficient <= 0:
            raise ValueError("Birkhoff selected non-positive coefficient")
        real_edges: list[tuple[int, int]] = []
        dummy_edges: list[tuple[int, int]] = []
        for source, destination, edge_kind, _ in selected:
            if edge_kind == "real":
                real[source][destination] -= coefficient
                real_edges.append((source, destination))
            else:
                dummy[source][destination] -= coefficient
                dummy_edges.append((source, destination))
        if not real_edges:
            raise ValueError("Birkhoff emitted an all-dummy interval")
        intervals.append(
            BirkhoffInterval(
                interval_id=len(intervals),
                coefficient=coefficient,
                real_edges=tuple(sorted(real_edges)),
                dummy_edges=tuple(sorted(dummy_edges)),
            )
        )
        remaining_degree -= coefficient
        total_coefficient += coefficient

    if any(value for row in real for value in row):
        raise ValueError("Birkhoff left real residual demand")
    coverage: dict[tuple[int, int], int] = {}
    matching_ok = True
    for interval in intervals:
        srcs = [src_rank for src_rank, _ in interval.real_edges]
        dsts = [dst_rank for _, dst_rank in interval.real_edges]
        matching_ok = matching_ok and len(srcs) == len(set(srcs)) and len(dsts) == len(set(dsts))
        for edge in interval.real_edges:
            coverage[edge] = coverage.get(edge, 0) + interval.coefficient
    expected = {
        (source, destination): original[source][destination]
        for source in range(n)
        for destination in range(n)
        if original[source][destination] > 0
    }
    coverage_ok = coverage == expected
    if total_coefficient != delta or not coverage_ok or not matching_ok:
        raise ValueError("Birkhoff certificate verification failed")
    payload = {
        "rank_count": n,
        "delta": delta,
        "total_coefficient": total_coefficient,
        "intervals": tuple(intervals),
    }
    return BirkhoffCertificate(
        rank_count=n,
        delta=delta,
        total_coefficient=total_coefficient,
        coverage_verified=coverage_ok,
        matching_constraints_verified=matching_ok,
        intervals=tuple(intervals),
        certificate_digest=stable_digest(payload),
    )


def order_birkhoff(
    tasks: Iterable[BirkhoffTask], *, rank_count: int
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], BirkhoffCertificate]:
    """Order canonical tasks through bucket-count matching decomposition.

    The deployable phase-local Birkhoff policy decomposes the *number of
    canonical buckets* on each edge, then pops one unchanged bucket from every
    real edge for each unit of an interval coefficient.  This is deliberately
    distinct from the fluid byte-volume reference used only for theoretical
    lower-bound analysis.
    """

    items = tuple(tasks)
    count_matrix = [[0 for _ in range(rank_count)] for _ in range(rank_count)]
    queues: dict[tuple[int, int], list[BirkhoffTask]] = {}
    for task in items:
        if not task.task_id or task.payload_units <= 0:
            raise ValueError("Birkhoff tasks require non-empty IDs and positive units")
        if task.src_rank == task.dst_rank:
            raise ValueError("local diagonal task must not enter Birkhoff DataPlane core")
        edge = (int(task.src_rank), int(task.dst_rank))
        count_matrix[edge[0]][edge[1]] += 1
        queues.setdefault(edge, []).append(task)
    for queue in queues.values():
        queue.sort(key=lambda item: (item.chunk_index, item.byte_offset, item.task_id))
    certificate = decompose_integer_matrix(count_matrix)
    waves: list[tuple[str, ...]] = []
    for interval in certificate.intervals:
        for _ in range(int(interval.coefficient)):
            selected: list[str] = []
            for edge in interval.real_edges:
                queue = queues.get(edge, [])
                if not queue:
                    raise ValueError("Birkhoff bucket decomposition over-served an edge")
                selected.append(queue.pop(0).task_id)
            if not selected:
                raise ValueError("Birkhoff bucket decomposition emitted an empty wave")
            waves.append(tuple(selected))
    if any(queue for queue in queues.values()):
        raise ValueError("Birkhoff bucket decomposition left canonical tasks unscheduled")
    order = tuple(task_id for wave in waves for task_id in wave)
    if len(order) != len(items) or len(set(order)) != len(items):
        raise ValueError("Birkhoff bucket-wave projection lost or duplicated a canonical task")
    return order, tuple(waves), certificate


__all__ = [
    "BirkhoffCertificate",
    "BirkhoffInterval",
    "BirkhoffTask",
    "decompose_integer_matrix",
    "order_birkhoff",
]
