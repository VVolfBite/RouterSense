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
    from rs.runtime.offline.exact_oracle_suite import (
        build_exact_problem,
        generate_exact_instances,
        solve_exact_instance,
    )
    from rs.runtime.offline.oracle_gap_replay import _run_policy

    policies = (
        "fifo_bucket",
        "birkhoff_bucket_phase_local",
        "current:p012:joint:event:rscf",
        "current:p012:joint:global:rscf",
    )
    for instance in generate_exact_instances(8):
        joint = solve_exact_instance(instance, scope="joint")
        assert joint["solver_status"] == "OPTIMAL"
        o_joint = float(joint["objective"])
        for policy_name in policies:
            row = _run_policy(build_exact_problem(instance), policy_name, instance)
            assert row["valid"] is True
            assert float(row["makespan"]) + 1.0e-9 >= o_joint

