from __future__ import annotations

from rs.scheduling import GlobalReadySetOptions, LogicalTopology, MultiPhaseSchedulingProblem, ReleaseConstraint, resolve_policy
from rs.scheduling.contracts import FlowDemand, FlowWindow, ForecastPressure


def _problem(*, p2_matrix=((0, 4), (6, 0))) -> MultiPhaseSchedulingProblem:
    p0 = ((0, 2), (8, 0))
    p1 = ((0, 10), (1, 0))
    p2 = tuple(tuple(int(value) for value in row) for row in p2_matrix)

    def _flows(matrix, phase, release_state, executable):
        flows = []
        for src, row in enumerate(matrix):
            for dst, byte_count in enumerate(row):
                if src == dst or int(byte_count) <= 0:
                    continue
                flows.append(
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
        return tuple(flows)

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
            digest="safe",
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(2, 2),
            matrix_total_bytes=sum(sum(row) for row in p2),
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


def test_safe_policy_falls_back_to_paired_b_when_raw_u_is_worse() -> None:
    problem = _problem()
    plan = resolve_policy(policy_name="RS_safe_gated_maxweight", bucket_rows=0).build_logical_plan(problem)
    assert plan.diagnostics["safe_policy"] == "RS_safe_gated_maxweight"
    assert plan.diagnostics["raw_u_policy"] == "U_gated_maxweight_matching"
    assert plan.diagnostics["paired_b_policy"] == "B_gated_maxweight_matching"
    assert plan.diagnostics["safe_makespan"] <= plan.diagnostics["paired_b_makespan"]


def test_safe_policy_keeps_raw_u_when_not_worse_than_paired_b() -> None:
    problem = _problem(p2_matrix=((0, 20), (1, 0)))
    plan = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0).build_logical_plan(problem)
    assert plan.diagnostics["safe_policy"] == "RS_safe_barrier_criticality"
    assert plan.diagnostics["safe_makespan"] <= plan.diagnostics["paired_b_makespan"]
    assert plan.diagnostics["same_information_guard"] is True
