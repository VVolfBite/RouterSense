from __future__ import annotations

import time
from dataclasses import replace

from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix
from rs.scheduling.validation import stable_hash

from .contracts import MatrixRows, ReconciledExecutionPlan, ReconciliationOutcome, TargetLayerPreparedJointPlan


from .future_prepared_reconcile import (
    is_future_prepared_plan, reconcile_future_prepared_order,
)

def _transpose(matrix: MatrixRows) -> MatrixRows:
    if not matrix:
        return ()
    width = len(matrix[0])
    return tuple(tuple(int(matrix[row][col]) for row in range(len(matrix))) for col in range(width))


def _edge_counts(matrix: MatrixRows) -> dict[tuple[int, int], int]:
    return {
        (src, dst): int(value)
        for src, row in enumerate(matrix)
        for dst, value in enumerate(row)
        if src != dst and int(value) > 0
    }


def _repair_logical_plan(
    *,
    prepared_plan: TargetLayerPreparedJointPlan,
    actual_p0_rows: MatrixRows,
) -> tuple[LogicalSchedulePlan, int, int, int, float]:
    actual_p0 = _edge_counts(canonicalize_remote_matrix(actual_p0_rows))
    actual_p1 = _edge_counts(canonicalize_remote_matrix(_transpose(actual_p0_rows)))
    expected_p0 = _edge_counts(canonicalize_remote_matrix(prepared_plan.h1_rows))
    expected_p1 = _edge_counts(canonicalize_remote_matrix(prepared_plan.derived_p1_rows))

    def _repair_phase(
        *,
        phase: str,
        expected: dict[tuple[int, int], int],
        actual: dict[tuple[int, int], int],
        inserted_prefix: str,
        inserted_release_state: str,
    ) -> tuple[list[FlowDemand], list[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
        matched = {edge for edge in expected if edge in actual}
        removed = {edge for edge in expected if edge not in actual}
        inserted = {edge for edge in actual if edge not in expected}
        resized = {edge for edge in matched if int(expected[edge]) != int(actual[edge])}
        replacement: list[FlowDemand] = []
        preferred_order: list[tuple[int, int]] = []
        emitted_edges: set[tuple[int, int]] = set()
        for wave in prepared_plan.logical_plan.waves:
            for flow in wave.flows:
                edge = (int(flow.src_rank), int(flow.dst_rank))
                if str(flow.phase) != phase or edge not in actual or edge in emitted_edges:
                    continue
                emitted_edges.add(edge)
                preferred_order.append(edge)
                replacement.append(replace(flow, byte_count=int(actual[edge])))
        for edge in sorted(inserted):
            replacement.append(
                FlowDemand(
                    flow_id=f"{inserted_prefix}:{edge[0]}->{edge[1]}",
                    phase=phase,
                    src_rank=int(edge[0]),
                    dst_rank=int(edge[1]),
                    byte_count=int(actual[edge]),
                    release_state=str(inserted_release_state),
                    is_executable=True,
                    dependency_metadata={"inserted_by_repair": True, "phase": phase},
                )
            )
        return replacement, preferred_order, matched, removed, resized

    p0_replacement, p0_preferred_order, p0_matched, p0_removed, p0_resized = _repair_phase(
        phase="p0_dispatch",
        expected=expected_p0,
        actual=actual_p0,
        inserted_prefix="repaired_p0",
        inserted_release_state="ready",
    )
    p1_replacement, p1_preferred_order, p1_matched, p1_removed, p1_resized = _repair_phase(
        phase="p1_return",
        expected=expected_p1,
        actual=actual_p1,
        inserted_prefix="repaired_p1",
        inserted_release_state="blocked",
    )

    replacement = [*p0_replacement, *p1_replacement]
    logical_plan = LogicalSchedulePlan(
        policy_name=str(prepared_plan.logical_plan.policy_name),
        waves=(LogicalWave(wave_id=0, flows=tuple(replacement), duration=0.0),),
        diagnostics={
            **dict(prepared_plan.logical_plan.diagnostics or {}),
            "prepared_repair": True,
            "prepared_repair_preserved_p0_order": [f"{src}->{dst}" for src, dst in p0_preferred_order],
            "prepared_repair_preserved_p1_order": [f"{src}->{dst}" for src, dst in p1_preferred_order],
            "logical_plan_digest": stable_hash([flow.to_dict() for flow in replacement]),
        },
    )
    expected_edge_count = len(expected_p0) + len(expected_p1)
    matched_edge_count = len(p0_matched) + len(p1_matched)
    removed_count = len(p0_removed) + len(p1_removed)
    resized_count = len(p0_resized) + len(p1_resized)
    inserted_count = max(0, len(p0_replacement) - len(p0_matched)) + max(0, len(p1_replacement) - len(p1_matched))
    preserved_ratio = 1.0 if expected_edge_count == 0 else float(matched_edge_count) / float(expected_edge_count)
    return logical_plan, inserted_count, removed_count, resized_count, preserved_ratio


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
            status="exact",
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
            status="rejected",
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
    status = "repaired"
    preserved = 1.0 if total_predicted_edges == 0 else max(0.0, float(matched_edges) / float(total_predicted_edges))
    if preserved == 0.0 and (removed_edges > 0 or new_edges > 0):
        status = "rejected"
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


def reconcile_once(
    *,
    prepared_plan: TargetLayerPreparedJointPlan,
    actual_p0_rows: MatrixRows,
    frozen_frontier: set[str] | None = None,
) -> ReconciledExecutionPlan:
    if is_future_prepared_plan(prepared_plan):
        return reconcile_future_prepared_order(
            prepared_plan=prepared_plan,
            actual_p0_rows=actual_p0_rows,
            frozen_frontier=frozen_frontier,
        )
    outcome = reconcile_target_plan(prepared_plan=prepared_plan, actual_p0_rows=actual_p0_rows)
    if outcome.status == "exact":
        return ReconciledExecutionPlan(
            status="exact",
            logical_plan=prepared_plan.logical_plan,
            logical_plan_digest=str(prepared_plan.logical_plan_digest),
            preserved_edge_ratio=1.0,
            inserted_edge_count=0,
            removed_edge_count=0,
            resized_edge_count=0,
            repair_us=float(outcome.repair_us),
            details={"frozen_frontier": sorted(frozen_frontier or ())},
        )
    if outcome.status == "rejected":
        return ReconciledExecutionPlan(
            status="rejected",
            logical_plan=None,
            logical_plan_digest=None,
            preserved_edge_ratio=float(outcome.preserved_order_ratio),
            inserted_edge_count=int(outcome.new_edges),
            removed_edge_count=int(outcome.removed_edges),
            resized_edge_count=int(outcome.resized_edges),
            repair_us=float(outcome.repair_us),
            details={**dict(outcome.details), "frozen_frontier": sorted(frozen_frontier or ())},
        )
    logical_plan, inserted_count, removed_count, resized_count, preserved_ratio = _repair_logical_plan(
        prepared_plan=prepared_plan,
        actual_p0_rows=actual_p0_rows,
    )
    return ReconciledExecutionPlan(
        status="repaired",
        logical_plan=logical_plan,
        logical_plan_digest=str(logical_plan.diagnostics.get("logical_plan_digest", stable_hash(logical_plan.to_dict()))),
        preserved_edge_ratio=float(preserved_ratio),
        inserted_edge_count=int(inserted_count),
        removed_edge_count=int(removed_count),
        resized_edge_count=int(resized_count),
        repair_us=float(outcome.repair_us),
        details={**dict(outcome.details), "frozen_frontier": sorted(frozen_frontier or ())},
    )
