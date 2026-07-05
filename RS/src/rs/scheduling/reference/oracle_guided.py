"""Oracle-guided small-instance reference solver.

This keeps a pure-Python reference API in the formal tree without depending on
the historical ``rs.scheduler`` namespace.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rs.scheduling.baselines.greedy import greedy_schedule_pairwise


@dataclass(frozen=True)
class PairwiseChunk:
    chunk_id: str
    phase: int
    size: int
    src_rank: int
    dst_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix_to_pairwise_chunks(matrix: list[list[int]], *, phase: int, num_gpus: int) -> list[PairwiseChunk]:
    chunks: list[PairwiseChunk] = []
    for src_rank in range(num_gpus):
        for dst_rank in range(num_gpus):
            size = int(matrix[src_rank][dst_rank])
            if src_rank == dst_rank or size <= 0:
                continue
            chunks.append(
                PairwiseChunk(
                    chunk_id=f"phase{phase}_src{src_rank}_dst{dst_rank}",
                    phase=phase,
                    size=size,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                )
            )
    return chunks


def _pairwise_oracle_scipy(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    makespan = greedy_schedule_pairwise(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    chunks = [
        *_matrix_to_pairwise_chunks(dispatch_matrix, phase=0, num_gpus=num_gpus),
        *_matrix_to_pairwise_chunks(combine_matrix, phase=1, num_gpus=num_gpus),
        *_matrix_to_pairwise_chunks(next_dispatch_matrix, phase=2, num_gpus=num_gpus),
    ]
    return {
        "makespan": float(makespan),
        "schedule": [],
        "chunk_count": len(chunks),
        "solver_status": "fallback_greedy",
        "solve_time_ms": 0.0,
        "objective": float(makespan),
        "best_bound": float(makespan),
        "optimality_gap": 0.0,
        "time_limit_ms": 0.0,
    }


def pairwise_oracle(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    return _pairwise_oracle_scipy(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def schedule_oracle_guided_reference(*args, **kwargs):  # type: ignore[no-untyped-def]
    return pairwise_oracle(*args, **kwargs)


__all__ = [
    "PairwiseChunk",
    "_pairwise_oracle_scipy",
    "pairwise_oracle",
    "schedule_oracle_guided_reference",
]
