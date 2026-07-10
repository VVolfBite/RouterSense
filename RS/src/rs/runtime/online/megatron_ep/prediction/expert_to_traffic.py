"""Map source-rank x expert counts into remote-only traffic matrices."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_col_sums_remote,
    matrix_diagonal_report,
    matrix_nonzero_remote_edge_count,
    matrix_remote_bytes,
    matrix_row_sums_remote,
    matrix_self_bytes,
)

from .expert_trace import SourceExpertCountMatrix


Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ExpertToTrafficAudit:
    actual_remote_bytes: int
    reconstructed_remote_bytes: int
    relative_l1_error: float
    cosine_similarity: float
    topk_edge_overlap: float
    row_sum_error: float
    col_sum_error: float
    self_bytes_ignored: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_expert_counts_to_traffic_matrix(
    counts: SourceExpertCountMatrix,
    expert_to_rank: Mapping[int, int],
    *,
    bytes_per_token: int,
    top_k: int | None = None,
) -> Matrix:
    world_size = int(counts.world_size)
    traffic = [[0 for _ in range(world_size)] for _ in range(world_size)]
    for source_rank, row in enumerate(counts.counts):
        for expert_id, count in enumerate(row):
            if int(expert_id) not in expert_to_rank:
                raise ValueError(f"missing expert_to_rank for expert_id={int(expert_id)}")
            dst_rank = int(expert_to_rank[int(expert_id)])
            if dst_rank < 0 or dst_rank >= world_size:
                raise ValueError("expert_to_rank must use EP-local rank indices")
            traffic[source_rank][dst_rank] += int(count) * int(bytes_per_token)
    return canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in traffic))


def compare_reconstructed_traffic(
    reconstructed_matrix: Matrix,
    actual_matrix: Matrix,
    *,
    topk: int = 16,
) -> ExpertToTrafficAudit:
    predicted = canonicalize_remote_matrix(reconstructed_matrix)
    actual = canonicalize_remote_matrix(actual_matrix)
    pred_values = [float(v) for row in predicted for v in row]
    act_values = [float(v) for row in actual for v in row]
    abs_l1 = float(sum(abs(a - b) for a, b in zip(pred_values, act_values, strict=False)))
    actual_total = float(sum(act_values))
    relative_l1 = 0.0 if actual_total <= 0.0 else abs_l1 / actual_total
    dot = float(sum(a * b for a, b in zip(pred_values, act_values, strict=False)))
    pred_norm = math.sqrt(sum(v * v for v in pred_values))
    act_norm = math.sqrt(sum(v * v for v in act_values))
    cosine = 0.0 if pred_norm == 0.0 or act_norm == 0.0 else dot / (pred_norm * act_norm)
    pred_edges = _sorted_edges(predicted)[: max(1, int(topk))]
    act_edges = _sorted_edges(actual)[: max(1, int(topk))]
    pred_edge_ids = {(src, dst) for src, dst, _ in pred_edges}
    act_edge_ids = {(src, dst) for src, dst, _ in act_edges}
    row_sum_error = float(sum(abs(a - b) for a, b in zip(matrix_row_sums_remote(predicted), matrix_row_sums_remote(actual), strict=False)))
    col_sum_error = float(sum(abs(a - b) for a, b in zip(matrix_col_sums_remote(predicted), matrix_col_sums_remote(actual), strict=False)))
    return ExpertToTrafficAudit(
        actual_remote_bytes=int(matrix_remote_bytes(actual)),
        reconstructed_remote_bytes=int(matrix_remote_bytes(predicted)),
        relative_l1_error=relative_l1,
        cosine_similarity=cosine,
        topk_edge_overlap=float(len(pred_edge_ids & act_edge_ids) / max(1, len(act_edge_ids))),
        row_sum_error=row_sum_error,
        col_sum_error=col_sum_error,
        self_bytes_ignored=int(matrix_self_bytes(reconstructed_matrix) + matrix_self_bytes(actual_matrix)),
    )


def reconstruction_report(reconstructed_matrix: Matrix, actual_matrix: Matrix) -> dict[str, Any]:
    return {
        "reconstructed_diagonal": matrix_diagonal_report(reconstructed_matrix),
        "actual_diagonal": matrix_diagonal_report(actual_matrix),
        "reconstructed_nonzero_edge_count": matrix_nonzero_remote_edge_count(reconstructed_matrix),
        "actual_nonzero_edge_count": matrix_nonzero_remote_edge_count(actual_matrix),
        "audit": compare_reconstructed_traffic(reconstructed_matrix, actual_matrix).to_dict(),
    }


def _sorted_edges(matrix: Matrix) -> list[tuple[int, int, int]]:
    edges = [
        (src, dst, int(value))
        for src, row in enumerate(matrix)
        for dst, value in enumerate(row)
        if src != dst and int(value) > 0
    ]
    edges.sort(key=lambda item: (-item[2], item[0], item[1]))
    return edges


__all__ = [
    "ExpertToTrafficAudit",
    "compare_reconstructed_traffic",
    "reconstruction_report",
    "source_expert_counts_to_traffic_matrix",
]
