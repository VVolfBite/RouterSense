from __future__ import annotations

from typing import List, Tuple

def greedy_schedule_single_layer(
    traffic_matrix: list[list[int]],
    num_gpus: int,
    gpu_earliest_start: list[float] | None = None,
) -> tuple[float, list[float]]:
    if gpu_earliest_start is None:
        gpu_earliest_start = [0.0] * num_gpus
    chunks: list[tuple[int, int, int]] = []
    for src in range(num_gpus):
        for dst in range(num_gpus):
            if src != dst and traffic_matrix[src][dst] > 0:
                chunks.append((traffic_matrix[src][dst], src, dst))
    chunks.sort(reverse=True)
    gpu_available = list(gpu_earliest_start)
    for size, src, dst in chunks:
        start = max(gpu_available[src], gpu_available[dst])
        end = start + float(size)
        gpu_available[src] = end
        gpu_available[dst] = end
    return max(gpu_available) if gpu_available else 0.0, gpu_available


def greedy_schedule_pairwise(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> float:
    gpu_available = [0.0] * num_gpus
    for matrix in (dispatch_matrix, combine_matrix, next_dispatch_matrix):
        chunks: list[tuple[int, int, int]] = []
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst and matrix[src][dst] > 0:
                    chunks.append((int(matrix[src][dst]), src, dst))
        chunks.sort(reverse=True)
        for size, src, dst in chunks:
            start = max(gpu_available[src], gpu_available[dst])
            end = start + float(size)
            gpu_available[src] = end
            gpu_available[dst] = end
    return max(gpu_available) if gpu_available else 0.0


def greedy_schedule_multi_layer(
    traffic_matrices: list[list[list[int]]],
    combine_matrices: list[list[list[int]]],
    num_gpus: int,
) -> float:
    gpu_earliest_start = [0.0] * num_gpus
    for dispatch_matrix, combine_matrix in zip(traffic_matrices, combine_matrices, strict=False):
        _, after_dispatch = greedy_schedule_single_layer(dispatch_matrix, num_gpus, gpu_earliest_start)
        _, after_combine = greedy_schedule_single_layer(combine_matrix, num_gpus, after_dispatch)
        gpu_earliest_start = after_combine
    return max(gpu_earliest_start) if gpu_earliest_start else 0.0
