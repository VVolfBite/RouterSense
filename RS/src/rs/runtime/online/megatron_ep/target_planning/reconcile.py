from __future__ import annotations

import time

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix

from .contracts import MatrixRows, ReconciliationOutcome, TargetLayerPreparedJointPlan


def _transpose(matrix: MatrixRows) -> MatrixRows:
    if not matrix:
        return ()
    width = len(matrix[0])
    return tuple(tuple(int(matrix[row][col]) for row in range(len(matrix))) for col in range(width))


def reconcile_target_plan(
    *,
    prepared_plan: TargetLayerPreparedJointPlan,
    actual_p0_rows: MatrixRows,
) -> ReconciliationOutcome:
    started = time.perf_counter_ns()
    expected = canonicalize_remote_matrix(prepared_plan.h1_rows)
    actual = canonicalize_remote_matrix(actual_p0_rows)
    if expected == actual:
        ended = time.perf_counter_ns()
        return ReconciliationOutcome(
            status="exact_match",
            matched_edges=sum(1 for i, row in enumerate(actual) for j, value in enumerate(row) if i != j and int(value) > 0),
            removed_edges=0,
            new_edges=0,
            resized_edges=0,
            preserved_order_ratio=1.0,
            repair_us=(ended - started) / 1000.0,
            result_h1_rows=actual,
            result_p1_rows=_transpose(actual),
            details={"target_layer_id": prepared_plan.target_layer_id},
        )
    removed_edges = 0
    new_edges = 0
    resized_edges = 0
    matched_edges = 0
    same_shape = len(expected) == len(actual) and {len(row) for row in expected} == {len(actual[0]) if actual else 0}
    if not same_shape:
        ended = time.perf_counter_ns()
        return ReconciliationOutcome(
            status="reject",
            matched_edges=0,
            removed_edges=0,
            new_edges=0,
            resized_edges=0,
            preserved_order_ratio=0.0,
            repair_us=(ended - started) / 1000.0,
            result_h1_rows=actual,
            result_p1_rows=_transpose(actual),
            details={"reason": "shape_mismatch"},
        )
    for src, row in enumerate(actual):
        for dst, actual_rows in enumerate(row):
            if src == dst:
                continue
            predicted_rows = int(expected[src][dst])
            actual_rows = int(actual_rows)
            if predicted_rows > 0 and actual_rows > 0:
                matched_edges += 1
                if predicted_rows != actual_rows:
                    resized_edges += 1
            elif predicted_rows > 0 and actual_rows == 0:
                removed_edges += 1
            elif predicted_rows == 0 and actual_rows > 0:
                new_edges += 1
    total_predicted_edges = sum(1 for i, row in enumerate(expected) for j, value in enumerate(row) if i != j and int(value) > 0)
    status = "repairable"
    preserved = 1.0 if total_predicted_edges == 0 else max(0.0, float(matched_edges) / float(total_predicted_edges))
    if preserved == 0.0 and (removed_edges > 0 or new_edges > 0):
        status = "reject"
    ended = time.perf_counter_ns()
    return ReconciliationOutcome(
        status=status,
        matched_edges=int(matched_edges),
        removed_edges=int(removed_edges),
        new_edges=int(new_edges),
        resized_edges=int(resized_edges),
        preserved_order_ratio=float(preserved),
        repair_us=(ended - started) / 1000.0,
        result_h1_rows=actual,
        result_p1_rows=_transpose(actual),
        details={"target_layer_id": prepared_plan.target_layer_id},
    )

