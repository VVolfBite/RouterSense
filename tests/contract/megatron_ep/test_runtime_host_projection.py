from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import RuntimeHostFeasibilityProjector, host_project_safe_selection
from rs.scheduling.contracts import LogicalSchedulePlan, LogicalWave


def _plan(name: str, makespan: float, raw_schedule: list[dict]) -> LogicalSchedulePlan:
    return LogicalSchedulePlan(
        policy_name=name,
        waves=(LogicalWave(wave_id=0, flows=(), duration=makespan),),
        diagnostics={"makespan": makespan, "raw_schedule": raw_schedule},
    )


def test_host_projection_preserves_rank_level_only_constraint() -> None:
    plan = _plan(
        "joint_example",
        10.0,
        [
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "end": 3.0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "end": 2.0},
        ],
    )
    projected = RuntimeHostFeasibilityProjector().project(plan)
    assert projected.host_projected_estimated_makespan >= projected.ideal_estimated_makespan or projected.host_projected_estimated_makespan >= 3.0
    assert "no_per_bucket_compute_overlap" in projected.projection_constraints


def test_host_project_safe_selection_compares_projected_plans() -> None:
    joint_candidate = _plan(
        "joint_example",
        8.0,
        [
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "end": 6.0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "end": 5.0},
        ],
    )
    local_fallback = _plan(
        "local_example",
        9.0,
        [
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "end": 4.0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "end": 7.0},
        ],
    )
    summary = host_project_safe_selection(joint_plan=joint_candidate, local_plan=local_fallback)
    assert "host_projected_safe_selection" in summary
    assert summary["projection_constraints"]


def test_host_projection_preserves_duration_when_release_is_delayed() -> None:
    plan = _plan(
        "joint_duration",
        3.0,
        [
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "start": 0.0, "end": 5.0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "start": 1.0, "end": 3.0},
        ],
    )
    projected = RuntimeHostFeasibilityProjector().project(plan)
    assert projected.host_projected_estimated_makespan == 7.0
