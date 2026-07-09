"""Compare predicted next-dispatch matrices against actual observed dispatch matrices."""

from __future__ import annotations

import math

from rs.scheduling.validation import stable_hash

from .contracts import Matrix, PredictionAuditRecord, PredictedTrafficMatrix


def compare_predicted_to_actual(
    predicted: PredictedTrafficMatrix,
    actual_matrix: Matrix,
    *,
    topk: int = 16,
) -> PredictionAuditRecord:
    if len(predicted.matrix) != len(actual_matrix) or any(len(a) != len(b) for a, b in zip(predicted.matrix, actual_matrix)):
        return PredictionAuditRecord(
            predictor_name=predicted.predictor_name,
            source_layer_id=predicted.source_layer_id,
            predicted_layer_id=predicted.predicted_layer_id,
            predicted_digest=predicted.matrix_digest,
            actual_digest=stable_hash({"matrix": actual_matrix}),
            predicted_total_bytes=int(predicted.total_bytes),
            actual_total_bytes=int(sum(sum(row) for row in actual_matrix)),
            relative_l1_error=0.0,
            absolute_l1_error=0.0,
            cosine_similarity=0.0,
            topk_edge_overlap=0.0,
            nonzero_edge_precision=0.0,
            nonzero_edge_recall=0.0,
            evaluation_eligible=bool(predicted.evaluation_eligible),
            valid=False,
            error="matrix_shape_mismatch",
        )

    predicted_values = [float(value) for row in predicted.matrix for value in row]
    actual_values = [float(value) for row in actual_matrix for value in row]
    absolute_l1_error = float(sum(abs(a - b) for a, b in zip(predicted_values, actual_values)))
    actual_total = float(sum(actual_values))
    relative_l1_error = float(absolute_l1_error / actual_total) if actual_total > 0.0 else 0.0
    predicted_norm = math.sqrt(sum(value * value for value in predicted_values))
    actual_norm = math.sqrt(sum(value * value for value in actual_values))
    dot = sum(a * b for a, b in zip(predicted_values, actual_values))
    cosine_similarity = float(dot / (predicted_norm * actual_norm)) if predicted_norm > 0.0 and actual_norm > 0.0 else 0.0

    predicted_edges = _sorted_edges(predicted.matrix)[: max(1, int(topk))]
    actual_edges = _sorted_edges(actual_matrix)[: max(1, int(topk))]
    predicted_edge_ids = {(src, dst) for src, dst, _ in predicted_edges}
    actual_edge_ids = {(src, dst) for src, dst, _ in actual_edges}
    topk_edge_overlap = float(len(predicted_edge_ids & actual_edge_ids) / max(1, len(actual_edge_ids)))

    predicted_nonzero = {(src, dst) for src, row in enumerate(predicted.matrix) for dst, value in enumerate(row) if src != dst and int(value) > 0}
    actual_nonzero = {(src, dst) for src, row in enumerate(actual_matrix) for dst, value in enumerate(row) if src != dst and int(value) > 0}
    precision = float(len(predicted_nonzero & actual_nonzero) / len(predicted_nonzero)) if predicted_nonzero else 0.0
    recall = float(len(predicted_nonzero & actual_nonzero) / len(actual_nonzero)) if actual_nonzero else 0.0

    return PredictionAuditRecord(
        predictor_name=predicted.predictor_name,
        source_layer_id=predicted.source_layer_id,
        predicted_layer_id=predicted.predicted_layer_id,
        predicted_digest=predicted.matrix_digest,
        actual_digest=stable_hash({"matrix": actual_matrix}),
        predicted_total_bytes=int(predicted.total_bytes),
        actual_total_bytes=int(actual_total),
        relative_l1_error=relative_l1_error,
        absolute_l1_error=absolute_l1_error,
        cosine_similarity=cosine_similarity,
        topk_edge_overlap=topk_edge_overlap,
        nonzero_edge_precision=precision,
        nonzero_edge_recall=recall,
        evaluation_eligible=bool(predicted.evaluation_eligible),
    )


def _sorted_edges(matrix: Matrix) -> list[tuple[int, int, int]]:
    edges = [
        (src, dst, int(value))
        for src, row in enumerate(matrix)
        for dst, value in enumerate(row)
        if src != dst and int(value) > 0
    ]
    edges.sort(key=lambda item: (-item[2], item[0], item[1]))
    return edges
