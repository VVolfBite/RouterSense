"""Expert-level prediction metrics and expert->traffic error decomposition."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_col_sums_remote,
    matrix_row_sums_remote,
)

from .expert_to_traffic import compare_reconstructed_traffic, source_expert_counts_to_traffic_matrix
from .expert_trace import SourceExpertCountMatrix


@dataclass(frozen=True)
class ExpertPredictionMetrics:
    expert_count_relative_l1_error: float
    expert_count_cosine_similarity: float
    expert_topk_overlap: float
    expert_nonzero_precision: float
    expert_nonzero_recall: float
    source_rank_expert_l1_error: float
    bottleneck_expert_match: bool
    bottleneck_source_rank_match: bool
    traffic_relative_l1_error: float
    traffic_cosine_similarity: float
    traffic_topk_edge_overlap: float
    row_sum_error: float
    col_sum_error: float
    bottleneck_rank_match: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_expert_prediction(
    predicted_counts: SourceExpertCountMatrix,
    actual_counts: SourceExpertCountMatrix,
    *,
    expert_to_rank: Mapping[int, int],
    bytes_per_token: int,
    topk: int = 16,
) -> ExpertPredictionMetrics:
    pred_values = [float(v) for row in predicted_counts.counts for v in row]
    act_values = [float(v) for row in actual_counts.counts for v in row]
    abs_l1 = float(sum(abs(a - b) for a, b in zip(pred_values, act_values, strict=False)))
    actual_total = float(sum(act_values))
    relative_l1 = 0.0 if actual_total <= 0.0 else abs_l1 / actual_total
    dot = float(sum(a * b for a, b in zip(pred_values, act_values, strict=False)))
    pred_norm = math.sqrt(sum(v * v for v in pred_values))
    act_norm = math.sqrt(sum(v * v for v in act_values))
    cosine = 0.0 if pred_norm == 0.0 or act_norm == 0.0 else dot / (pred_norm * act_norm)
    pred_topk = _sorted_experts(predicted_counts)[: max(1, int(topk))]
    act_topk = _sorted_experts(actual_counts)[: max(1, int(topk))]
    pred_topk_ids = {(src, expert_id) for src, expert_id, _ in pred_topk}
    act_topk_ids = {(src, expert_id) for src, expert_id, _ in act_topk}
    pred_nonzero = {(src, expert_id) for src, row in enumerate(predicted_counts.counts) for expert_id, value in enumerate(row) if int(value) > 0}
    act_nonzero = {(src, expert_id) for src, row in enumerate(actual_counts.counts) for expert_id, value in enumerate(row) if int(value) > 0}
    source_rank_l1 = []
    for pred_row, act_row in zip(predicted_counts.counts, actual_counts.counts, strict=False):
        source_rank_l1.append(sum(abs(float(a) - float(b)) for a, b in zip(pred_row, act_row, strict=False)))
    predicted_matrix = source_expert_counts_to_traffic_matrix(predicted_counts, expert_to_rank, bytes_per_token=bytes_per_token)
    actual_matrix = source_expert_counts_to_traffic_matrix(actual_counts, expert_to_rank, bytes_per_token=bytes_per_token)
    traffic_audit = compare_reconstructed_traffic(predicted_matrix, actual_matrix, topk=topk)
    pred_row_sums = matrix_row_sums_remote(predicted_matrix)
    act_row_sums = matrix_row_sums_remote(actual_matrix)
    pred_col_sums = matrix_col_sums_remote(predicted_matrix)
    act_col_sums = matrix_col_sums_remote(actual_matrix)
    return ExpertPredictionMetrics(
        expert_count_relative_l1_error=relative_l1,
        expert_count_cosine_similarity=cosine,
        expert_topk_overlap=float(len(pred_topk_ids & act_topk_ids) / max(1, len(act_topk_ids))),
        expert_nonzero_precision=float(len(pred_nonzero & act_nonzero) / max(1, len(pred_nonzero))),
        expert_nonzero_recall=float(len(pred_nonzero & act_nonzero) / max(1, len(act_nonzero))),
        source_rank_expert_l1_error=float(sum(source_rank_l1) / max(1, len(source_rank_l1))),
        bottleneck_expert_match=_top_expert(predicted_counts) == _top_expert(actual_counts),
        bottleneck_source_rank_match=_top_source_rank(predicted_counts) == _top_source_rank(actual_counts),
        traffic_relative_l1_error=float(traffic_audit.relative_l1_error),
        traffic_cosine_similarity=float(traffic_audit.cosine_similarity),
        traffic_topk_edge_overlap=float(traffic_audit.topk_edge_overlap),
        row_sum_error=float(sum(abs(float(a) - float(b)) for a, b in zip(pred_row_sums, act_row_sums, strict=False))),
        col_sum_error=float(sum(abs(float(a) - float(b)) for a, b in zip(pred_col_sums, act_col_sums, strict=False))),
        bottleneck_rank_match=_top_rank(predicted_matrix) == _top_rank(actual_matrix),
    )


def _sorted_experts(counts: SourceExpertCountMatrix) -> list[tuple[int, int, int]]:
    rows = [
        (src, expert_id, int(value))
        for src, row in enumerate(counts.counts)
        for expert_id, value in enumerate(row)
        if int(value) > 0
    ]
    rows.sort(key=lambda item: (-item[2], item[0], item[1]))
    return rows


def _top_expert(counts: SourceExpertCountMatrix) -> int | None:
    totals = [0 for _ in range(counts.num_experts)]
    for row in counts.counts:
        for idx, value in enumerate(row):
            totals[idx] += int(value)
    if not any(totals):
        return None
    return int(max(range(len(totals)), key=lambda idx: totals[idx]))


def _top_source_rank(counts: SourceExpertCountMatrix) -> int | None:
    if not counts.counts:
        return None
    totals = [sum(int(v) for v in row) for row in counts.counts]
    if not any(totals):
        return None
    return int(max(range(len(totals)), key=lambda idx: totals[idx]))


def _top_rank(matrix: tuple[tuple[int, ...], ...]) -> int | None:
    canonical = canonicalize_remote_matrix(matrix)
    if not canonical:
        return None
    totals = [sum(int(v) for v in row) for row in canonical]
    if not any(totals):
        return None
    return int(max(range(len(totals)), key=lambda idx: totals[idx]))


__all__ = ["ExpertPredictionMetrics", "evaluate_expert_prediction"]
