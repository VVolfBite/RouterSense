from __future__ import annotations

from rs.scheduling import ForecastPressure, FlowDemand, FlowWindow, GlobalReadySetOptions, LogicalTopology, MultiPhaseSchedulingProblem, ReleaseConstraint, resolve_policy
from rs.runtime.online.megatron_ep.async_release import simulate_async_release


def _flows(matrix, phase, release_state, executable):
    result = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            result.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    byte_count=int(byte_count),
                    release_state=release_state,
                    is_executable=executable,
                )
            )
    return tuple(result)


def _problem() -> MultiPhaseSchedulingProblem:
    p0 = (
        (0, 2, 8, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (8, 0, 0, 0),
    )
    p1 = (
        (0, 0, 0, 0),
        (0, 0, 0, 9),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    p2 = (
        (0, 4, 6, 0),
        (0, 0, 0, 2),
        (0, 0, 0, 0),
        (3, 0, 0, 0),
    )
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(p0, "p0_dispatch", "ready", True),
            blocked_flows=_flows(p1, "p1_return", "blocked", False),
            forecast_pressure=_flows(p2, "p2_next_dispatch_forecast", "advisory_only", False),
        ),
        topology=LogicalTopology(num_gpus=4),
        release_model=ReleaseConstraint(phase="p1_return", rank=0, release_after_phase="p0_dispatch", expert_compute_delay=0.0),
        forecast=ForecastPressure(
            source="copy_current_dispatch",
            digest="f0",
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(4, 4),
            matrix_total_bytes=15,
            matrix=p2,
        ),
        options=GlobalReadySetOptions(
            scheduling_mode="runtime_lookahead",
            information_mode="p0_p1_p2",
            prediction_confidence=1.0,
        ),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def test_joint_bridge_policy_exposes_online_metadata() -> None:
    policy = resolve_policy(policy_name="routersense_joint_priority_phase_sync", bucket_rows=0)
    plan = policy.build_logical_plan(_problem())
    assert plan.diagnostics["policy_name"] == "routersense_joint_priority_phase_sync"
    assert plan.diagnostics["service_model"] == "phase_sync_joint_priority"
    assert plan.diagnostics["p2_source"] == "copy_current_dispatch"


def test_async_release_joint_bridge_path_can_beat_phase_sync_birkhoff_on_sensitive_case() -> None:
    birkhoff = resolve_policy(policy_name="birkhoff_phase_local", bucket_rows=0).build_logical_plan(_problem())
    sim = simulate_async_release(
        p0_dispatch_matrix=(
            (0, 2, 8, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (8, 0, 0, 0),
        ),
        p1_return_matrix=(
            (0, 0, 0, 0),
            (0, 0, 0, 9),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        predicted_p2_matrix=(
            (0, 4, 6, 0),
            (0, 0, 0, 2),
            (0, 0, 0, 0),
            (3, 0, 0, 0),
        ),
        compute_delay=0.0,
        policy_name="routersense_joint_async_release",
    )
    assert float(sim["completion_time"]) < float(birkhoff.diagnostics["makespan"])
