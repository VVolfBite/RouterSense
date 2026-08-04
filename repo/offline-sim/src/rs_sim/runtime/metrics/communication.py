from __future__ import annotations

"""Communication-only metric helpers.

These helpers intentionally operate on causal intervals rather than subtracting
one coarse compute span from one coarse communication span.  They therefore
preserve communication/compute overlap and remove only periods in which no
communication task is ready or in flight.
"""

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def _validate_time_ns(value: int, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def interval_union_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Return the measure of the union of half-open nanosecond intervals.

    Adjacent intervals are merged.  Zero-duration intervals are ignored.  This
    is the foundation for both the virtual communication clock and the network
    active-time metric.
    """

    normalized: list[tuple[int, int]] = []
    for index, (start, finish) in enumerate(intervals):
        start_ns = _validate_time_ns(start, name=f"interval[{index}].start")
        finish_ns = _validate_time_ns(finish, name=f"interval[{index}].finish")
        if finish_ns < start_ns:
            raise ValueError("interval finish must not precede start")
        if finish_ns == start_ns:
            continue
        normalized.append((start_ns, finish_ns))
    if not normalized:
        return 0
    normalized.sort()
    total = 0
    current_start, current_finish = normalized[0]
    for start_ns, finish_ns in normalized[1:]:
        if start_ns <= current_finish:
            current_finish = max(current_finish, finish_ns)
            continue
        total += current_finish - current_start
        current_start, current_finish = start_ns, finish_ns
    total += current_finish - current_start
    return int(total)


def compute_excluded_communication_makespan_ns(
    *,
    task_ready_complete_intervals: Iterable[tuple[int, int]],
) -> int:
    """Measure the global virtual communication clock.

    A task contributes from the instant it is causally ready until it completes.
    Consequently, the clock advances while at least one task is ready-but-not-
    completed (including in-flight work) and pauses only when every remaining
    task is still blocked on computation or another causal dependency.
    """

    return interval_union_duration_ns(task_ready_complete_intervals)


def network_active_union_ns(
    *,
    task_start_complete_intervals: Iterable[tuple[int, int]],
) -> int:
    """Return wall-clock time during which at least one transfer is in flight."""

    return interval_union_duration_ns(task_start_complete_intervals)


def nearest_rank_percentile_ns(values: Sequence[int], percentile: int) -> int:
    """Return a deterministic nearest-rank percentile for rank-local evidence."""

    if not isinstance(percentile, int) or isinstance(percentile, bool):
        raise ValueError("percentile must be an integer")
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be in [0, 100]")
    rows = sorted(_validate_time_ns(value, name="rank value") for value in values)
    if not rows:
        return 0
    if percentile == 0:
        return rows[0]
    index = min(len(rows) - 1, max(0, math.ceil(percentile * len(rows) / 100) - 1))
    return int(rows[index])


@dataclass(frozen=True, slots=True)
class RankCommunicationExposureSummary:
    values_ns: tuple[int, ...]
    total_ns: int
    mean_ns: int
    max_ns: int
    p95_ns: int
    p99_ns: int
    critical_rank: int | None


def summarize_rank_communication_exposure_ns(
    values: Sequence[int],
) -> RankCommunicationExposureSummary:
    rows = tuple(_validate_time_ns(value, name="rank exposure") for value in values)
    if not rows:
        return RankCommunicationExposureSummary(
            values_ns=(),
            total_ns=0,
            mean_ns=0,
            max_ns=0,
            p95_ns=0,
            p99_ns=0,
            critical_rank=None,
        )
    maximum = max(rows)
    return RankCommunicationExposureSummary(
        values_ns=rows,
        total_ns=sum(rows),
        mean_ns=int(round(sum(rows) / len(rows))),
        max_ns=maximum,
        p95_ns=nearest_rank_percentile_ns(rows, 95),
        p99_ns=nearest_rank_percentile_ns(rows, 99),
        critical_rank=next(index for index, value in enumerate(rows) if value == maximum),
    )
