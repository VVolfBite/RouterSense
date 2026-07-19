from __future__ import annotations

from rs.runtime.offline.exact_oracle_suite import run_exact_scope_suite
from rs.scheduling.reference.exact_small_instance import EXACT_REFERENCE_MODEL_ID


def test_unified_exact_scope_suite_is_certified_and_scope_only() -> None:
    payload = run_exact_scope_suite(8)
    assert payload["instance_count"] == 8
    assert payload["comparable_count"] == 8
    assert payload["dominance_violation_count"] == 0
    assert all(row["reference_model"] == EXACT_REFERENCE_MODEL_ID for row in payload["rows"])
    assert all(row["oracle_local_status"] == "OPTIMAL" for row in payload["rows"])
    assert all(row["oracle_joint_status"] == "OPTIMAL" for row in payload["rows"])


def test_single_phase_exact_unsupported_scale_reports_task_count() -> None:
    from rs.scheduling.contracts import FlowDemand
    from rs.scheduling.reference.exact_small_instance import (
        MAX_BUCKET_TASK_COUNT,
        solve_exact_small_instance,
    )

    flows = tuple(
        FlowDemand(
            flow_id=f"flow-{index}",
            phase="p0_dispatch",
            src_rank=0,
            dst_rank=1,
            byte_count=1,
            release_state="ready",
            is_executable=True,
        )
        for index in range(MAX_BUCKET_TASK_COUNT + 1)
    )
    result = solve_exact_small_instance(flows=flows, rank_count=2)
    assert result["solver_status"] == "unsupported_scale"
    assert result["task_count"] == len(flows)


def test_same_model_heuristics_never_beat_certified_joint_oracle() -> None:
    from rs.runtime.offline.prediction_oracle_baseline_closure import _run_exact_oracle_suite

    rows, _summary = _run_exact_oracle_suite(8)
    policy_columns = (
        "phase_barrier_fifo",
        "birkhoff_phase_local",
        "joint_zero_hint",
        "joint_copy_current",
        "joint_perfect_trace_hint",
        "safe_copy_current",
        "safe_perfect_trace_hint",
    )
    for row in rows:
        o_joint = float(row["O_joint"])
        assert all(float(row[column]) + 1.0e-9 >= o_joint for column in policy_columns)
