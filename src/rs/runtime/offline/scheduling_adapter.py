"""Formal offline scheduling replay helpers.

This module owns the reusable replay surface that was historically hosted
under ``experiments.paper``.  Formal tests and runtime code must not import the
experiments package.
"""

from __future__ import annotations

import time
from typing import Any

from rs.runtime.offline.replay_unified import PlanningHint, ReplayEngine, ReplayWindow


def replay_window_from_matrices(
    *,
    fixture_id: str,
    layer_id: int,
    p0_matrix: tuple[tuple[int, ...], ...],
    p1_matrix: tuple[tuple[int, ...], ...],
    p2_matrix: tuple[tuple[int, ...], ...],
) -> ReplayWindow:
    return ReplayWindow(
        fixture_id=str(fixture_id),
        window_id=f"{layer_id}->{layer_id + 1}",
        layer_id=int(layer_id),
        p0_truth_rows=p0_matrix,
        p1_truth_rows=p1_matrix,
        p2_truth_rows=p2_matrix,
        matrix_unit="rows",
        group_size=len(p0_matrix),
        payload_row_bytes_by_phase={"P0": 1, "P1": 1, "P2": 1},
        metadata={},
    )


def execute_policy(
    *,
    replay_window: ReplayWindow,
    policy_name: str,
    hint_type: str,
    p2_hint_rows: tuple[tuple[int, ...], ...],
    confidence: float = 1.0,
    expert_compute_delay: float = 0.0,
    bucket_rows: int = 1,
    max_waves: int = 256,
) -> dict[str, Any]:
    """Execute one formal policy on an immutable replay window.

    Algorithm-catalog presentation metadata is intentionally not resolved
    here; experiment/reporting wrappers may add it without creating a reverse
    dependency from formal runtime code to experiment surfaces.
    """

    engine = ReplayEngine(
        scheduling_mode="execution_window",
        expert_compute_delay=float(expert_compute_delay),
        bucket_rows=int(bucket_rows),
        max_waves=int(max_waves),
    )
    hint = PlanningHint(
        hint_type=str(hint_type),
        p2_hint_rows=p2_hint_rows,
        confidence=float(confidence),
        source_layer=int(replay_window.layer_id),
        target_layer=int(replay_window.layer_id) + 1,
    )
    started = time.perf_counter_ns()
    result = engine.execute(
        replay_window=replay_window,
        planning_hint=hint,
        policy_name=str(policy_name),
    )
    ended = time.perf_counter_ns()
    result["planning_runtime_ms_wall"] = (ended - started) / 1_000_000.0
    return result


__all__ = ["execute_policy", "replay_window_from_matrices"]
