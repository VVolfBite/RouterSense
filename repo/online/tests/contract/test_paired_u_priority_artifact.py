from __future__ import annotations

from rs.scheduling import GlobalReadySetOptions, LogicalTopology, MultiPhaseSchedulingProblem, ReleaseConstraint, resolve_policy
from rs.scheduling.contracts import FlowDemand, FlowWindow, ForecastPressure
from rs.scheduling.online_adapters import build_priority_artifact_from_plan


def _problem() -> MultiPhaseSchedulingProblem:
    p0 = ((0, 4), (8, 0))
    p1 = ((0, 6), (2, 0))
    p2 = ((0, 3), (5, 0))

    def _flows(matrix, phase, release_state, executable):
        rows = []
        for src, row in enumerate(matrix):
            for dst, byte_count in enumerate(row):
                if src == dst or int(byte_count) <= 0:
                    continue
                rows.append(
                    FlowDemand(
                        flow_id=f"{phase}:{src}->{dst}",
                        phase=phase,
                        src_rank=src,
                        dst_rank=dst,
                        byte_count=int(byte_count),
                        release_state=release_state,
                        is_executable=executable,
                    )
                )
        return tuple(rows)

    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(p0, "p0_dispatch", "ready", True),
            blocked_flows=_flows(p1, "p1_return", "blocked", False),
            forecast_pressure=_flows(p2, "p2_next_dispatch_forecast", "advisory_only", False),
        ),
        topology=LogicalTopology(num_gpus=2),
        release_model=ReleaseConstraint(phase="p1_return", rank=0, release_after_phase="p0_dispatch", expert_compute_delay=0.0),
        forecast=ForecastPressure(
            source="copy_current_dispatch",
            digest="artifact",
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(2, 2),
            matrix_total_bytes=8,
            matrix=p2,
        ),
        options=GlobalReadySetOptions(scheduling_mode="runtime_lookahead", information_mode="p0_p1_p2", prediction_confidence=1.0),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def test_priority_artifact_contains_guard_metadata_and_entries() -> None:
    problem = _problem()
    plan = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0).build_logical_plan(problem)
    artifact = build_priority_artifact_from_plan(
        problem=problem,
        plan=plan,
        heuristic_family="barrier_criticality_matching",
        predictor_name="copy_current_dispatch",
        p2_source="copy_current_dispatch",
    )
    assert artifact.source_safe_policy == "RS_safe_barrier_criticality"
    assert artifact.heuristic_family == "barrier_criticality_matching"
    assert artifact.granularity_mode == "dynamic_bucket_current"
    assert artifact.priority_entries
    assert artifact.priority_digest
