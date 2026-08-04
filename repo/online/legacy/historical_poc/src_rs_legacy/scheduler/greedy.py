from __future__ import annotations


def _matrix_to_chunks(
    traffic_matrix: list[list[int]],
    num_gpus: int,
) -> list[tuple[int, int, int]]:
    chunks: list[tuple[int, int, int]] = []
    for src in range(num_gpus):
        for dst in range(num_gpus):
            if src == dst:
                continue
            size = int(traffic_matrix[src][dst])
            if size <= 0:
                continue
            chunks.append((size, src, dst))
    chunks.sort(reverse=True)
    return chunks


def greedy_schedule_single_layer(
    traffic_matrix: list[list[int]],
    num_gpus: int,
    gpu_earliest_start: list[float] | None = None,
    *,
    model: str = "full_duplex",
) -> tuple[float, list[float]]:
    if gpu_earliest_start is None:
        gpu_earliest_start = [0.0] * num_gpus
    chunks = _matrix_to_chunks(traffic_matrix, num_gpus)

    if model == "half_duplex":
        gpu_available = list(gpu_earliest_start)
        for size, src, dst in chunks:
            start = max(gpu_available[src], gpu_available[dst])
            end = start + float(size)
            gpu_available[src] = end
            gpu_available[dst] = end
        return max(gpu_available) if gpu_available else 0.0, gpu_available

    if model == "full_duplex":
        sender_available = list(gpu_earliest_start)
        receiver_available = list(gpu_earliest_start)
        for size, src, dst in chunks:
            start = max(sender_available[src], receiver_available[dst])
            end = start + float(size)
            sender_available[src] = end
            receiver_available[dst] = end
        all_ends = sender_available + receiver_available
        return max(all_ends) if all_ends else 0.0, [max(sender_available[g], receiver_available[g]) for g in range(num_gpus)]

    if model == "incast_only":
        receiver_available = list(gpu_earliest_start)
        for size, _src, dst in chunks:
            start = receiver_available[dst]
            end = start + float(size)
            receiver_available[dst] = end
        return max(receiver_available) if receiver_available else 0.0, receiver_available

    raise ValueError(f"Unsupported scheduling model: {model}")


def greedy_schedule_pairwise(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> float:
    matrices = (dispatch_matrix, combine_matrix, next_dispatch_matrix)

    if model == "half_duplex":
        gpu_available = [0.0] * num_gpus
        dispatch_receivers = {dst for _size, _src, dst in _matrix_to_chunks(dispatch_matrix, num_gpus)}
        for phase_index, matrix in enumerate(matrices):
            if phase_index == 1 and expert_compute_delay > 0.0:
                gpu_available = [
                    value + expert_compute_delay if gpu in dispatch_receivers else value
                    for gpu, value in enumerate(gpu_available)
                ]
            for size, src, dst in _matrix_to_chunks(matrix, num_gpus):
                start = max(gpu_available[src], gpu_available[dst])
                end = start + float(size)
                gpu_available[src] = end
                gpu_available[dst] = end
        return max(gpu_available) if gpu_available else 0.0

    if model == "full_duplex":
        sender_available = [0.0] * num_gpus
        receiver_available = [0.0] * num_gpus
        phase_receiver_done: dict[int, list[float]] = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus}
        for phase_index, matrix in enumerate(matrices):
            for size, src, dst in _matrix_to_chunks(matrix, num_gpus):
                start = max(sender_available[src], receiver_available[dst])
                if phase_index > 0:
                    delay = expert_compute_delay if phase_index == 1 else 0.0
                    start = max(start, phase_receiver_done[phase_index - 1][src] + delay)
                end = start + float(size)
                sender_available[src] = end
                receiver_available[dst] = end
                if phase_index in phase_receiver_done:
                    phase_receiver_done[phase_index][dst] = max(phase_receiver_done[phase_index][dst], end)
        return max(sender_available + receiver_available) if num_gpus else 0.0

    if model == "incast_only":
        receiver_available = [0.0] * num_gpus
        phase_receiver_done: dict[int, list[float]] = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus}
        for phase_index, matrix in enumerate(matrices):
            for size, src, dst in _matrix_to_chunks(matrix, num_gpus):
                start = receiver_available[dst]
                if phase_index > 0:
                    delay = expert_compute_delay if phase_index == 1 else 0.0
                    start = max(start, phase_receiver_done[phase_index - 1][src] + delay)
                end = start + float(size)
                receiver_available[dst] = end
                if phase_index in phase_receiver_done:
                    phase_receiver_done[phase_index][dst] = max(phase_receiver_done[phase_index][dst], end)
        return max(receiver_available) if receiver_available else 0.0

    raise ValueError(f"Unsupported scheduling model: {model}")


def greedy_schedule_multi_layer(
    traffic_matrices: list[list[list[int]]],
    combine_matrices: list[list[list[int]]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> float:
    gpu_earliest_start = [0.0] * num_gpus
    for dispatch_matrix, combine_matrix in zip(traffic_matrices, combine_matrices, strict=False):
        _, after_dispatch = greedy_schedule_single_layer(
            dispatch_matrix,
            num_gpus,
            gpu_earliest_start,
            model=model,
        )
        if expert_compute_delay > 0.0:
            after_dispatch = [value + expert_compute_delay for value in after_dispatch]
        _, after_combine = greedy_schedule_single_layer(
            combine_matrix,
            num_gpus,
            after_dispatch,
            model=model,
        )
        gpu_earliest_start = after_combine
    return max(gpu_earliest_start) if gpu_earliest_start else 0.0
