"""Dependency/release helpers for multiphase scheduling."""

from __future__ import annotations

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_row_sums_remote

from .flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE, ResidualFlowState


def collect_real_flows(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    *,
    mode: str,
) -> list[ResidualFlowState]:
    matrices = [dispatch_matrix, combine_matrix]
    if mode == EXECUTION_WINDOW_MODE:
        matrices.append(next_dispatch_matrix)
    elif mode != RUNTIME_LOOKAHEAD_MODE:
        raise ValueError(f"unsupported mode {mode!r}")
    flows: list[ResidualFlowState] = []
    for phase, matrix in enumerate(matrices):
        for src, row in enumerate(matrix):
            for dst, value in enumerate(row):
                size = float(value)
                if src == dst or size <= 0.0:
                    continue
                flows.append(
                    ResidualFlowState(
                        flow_id=f"phase{phase}_src{src}_dst{dst}",
                        phase=phase,
                        src_gpu=src,
                        dst_gpu=dst,
                        volume=size,
                    )
                )
    return flows


def outbound_loads(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    *,
    mode: str,
    prediction_confidence: float,
) -> dict[tuple[int, int], float]:
    matrices = [dispatch_matrix, combine_matrix, next_dispatch_matrix]
    loads: dict[tuple[int, int], float] = {}
    for phase, matrix in enumerate(matrices):
        scale = 1.0
        if phase == 2 and mode == RUNTIME_LOOKAHEAD_MODE:
            scale = max(0.0, min(1.0, prediction_confidence))
        remote_rows = matrix_row_sums_remote(canonicalize_remote_matrix(matrix))
        for gpu, row_sum in enumerate(remote_rows):
            loads[(phase, gpu)] = scale * float(row_sum)
    return loads


def inbound_remaining(flows: list[ResidualFlowState], num_gpus: int) -> dict[tuple[int, int], float]:
    remaining = {(phase, gpu): 0.0 for phase in range(3) for gpu in range(num_gpus)}
    for flow in flows:
        remaining[(flow.phase, flow.dst_gpu)] += float(flow.volume)
    return remaining
