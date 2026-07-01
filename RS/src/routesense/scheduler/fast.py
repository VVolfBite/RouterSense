from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledChunk:
    phase: int
    size: int
    src_gpu: int
    dst_gpu: int
    start: float
    end: float
    priority: tuple[float, ...]


def _matrix_chunks(
    matrix: list[list[int]],
    *,
    phase: int,
    num_gpus: int,
) -> list[tuple[int, int, int]]:
    chunks: list[tuple[int, int, int]] = []
    for src in range(num_gpus):
        for dst in range(num_gpus):
            size = int(matrix[src][dst])
            if src == dst or size <= 0:
                continue
            chunks.append((size, src, dst))
    return chunks


def _schedule_ordered_chunks(
    chunks: list[tuple[int, int, int]],
    *,
    phase: int,
    num_gpus: int,
    gpu_available: list[float],
    priority_fn,
) -> tuple[list[float], list[ScheduledChunk]]:
    ordered = sorted(chunks, key=priority_fn, reverse=True)
    scheduled: list[ScheduledChunk] = []
    for size, src, dst in ordered:
        start = max(gpu_available[src], gpu_available[dst])
        end = start + float(size)
        gpu_available[src] = end
        gpu_available[dst] = end
        scheduled.append(
            ScheduledChunk(
                phase=phase,
                size=size,
                src_gpu=src,
                dst_gpu=dst,
                start=start,
                end=end,
                priority=tuple(float(value) for value in priority_fn((size, src, dst))),
            )
        )
    return gpu_available, scheduled


def fast_schedule_pairwise(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> dict[str, object]:
    """
    Weighted look-ahead scheduler.

    Phase 0 prioritizes chunks whose destination becomes hot in the next-layer
    dispatch. Phase 1 prioritizes chunks whose return destination becomes hot as
    a sender in the next-layer dispatch. Phase 2 falls back to LPT.
    """
    next_inbound = [sum(int(next_dispatch_matrix[src][dst]) for src in range(num_gpus)) for dst in range(num_gpus)]
    next_outbound = [sum(int(next_dispatch_matrix[src][dst]) for dst in range(num_gpus)) for src in range(num_gpus)]
    gpu_available = [0.0] * num_gpus
    schedule: list[ScheduledChunk] = []

    dispatch_chunks = _matrix_chunks(dispatch_matrix, phase=0, num_gpus=num_gpus)
    combine_chunks = _matrix_chunks(combine_matrix, phase=1, num_gpus=num_gpus)
    next_chunks = _matrix_chunks(next_dispatch_matrix, phase=2, num_gpus=num_gpus)

    gpu_available, dispatch_schedule = _schedule_ordered_chunks(
        dispatch_chunks,
        phase=0,
        num_gpus=num_gpus,
        gpu_available=gpu_available,
        priority_fn=lambda item: (next_inbound[item[2]], item[0], -item[1], -item[2]),
    )
    schedule.extend(dispatch_schedule)

    gpu_available, combine_schedule = _schedule_ordered_chunks(
        combine_chunks,
        phase=1,
        num_gpus=num_gpus,
        gpu_available=gpu_available,
        priority_fn=lambda item: (next_outbound[item[2]], item[0], -item[1], -item[2]),
    )
    schedule.extend(combine_schedule)

    gpu_available, next_schedule = _schedule_ordered_chunks(
        next_chunks,
        phase=2,
        num_gpus=num_gpus,
        gpu_available=gpu_available,
        priority_fn=lambda item: (item[0], -item[1], -item[2]),
    )
    schedule.extend(next_schedule)

    return {
        "makespan": max(gpu_available) if gpu_available else 0.0,
        "schedule": [
            {
                "phase": chunk.phase,
                "size": chunk.size,
                "src_gpu": chunk.src_gpu,
                "dst_gpu": chunk.dst_gpu,
                "start": chunk.start,
                "end": chunk.end,
                "priority": list(chunk.priority),
            }
            for chunk in schedule
        ],
        "phase0_dst_hotness": next_inbound,
        "phase1_dst_hotness": next_outbound,
    }
