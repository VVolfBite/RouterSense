from __future__ import annotations

import json
from pathlib import Path

from rs.scheduling import GlobalReadySetOptions, LogicalTopology, MultiPhaseSchedulingProblem, ReleaseConstraint, resolve_policy
from rs.runtime.offline.runner import replay_and_audit_logical_plan, summarize_schedule_tail_metrics


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "scheduling"


def _load_problem(name: str, *, p2_source: str = "copy_current_dispatch") -> MultiPhaseSchedulingProblem:
    payload = json.loads((FIXTURE_ROOT / f"{name}.json").read_text(encoding="utf-8"))
    matrix0 = tuple(tuple(int(v) for v in row) for row in payload["p0_dispatch_matrix"])
    matrix1 = tuple(tuple(int(v) for v in row) for row in payload["p1_return_matrix"])
    matrix2 = tuple(tuple(int(v) for v in row) for row in payload["p2_next_dispatch_forecast_matrix"])
    from rs.scheduling.contracts import FlowDemand, FlowWindow, ForecastPressure

    def _flows(matrix, phase, release_state, executable):
        flows = []
        for src_rank, row in enumerate(matrix):
            for dst_rank, byte_count in enumerate(row):
                if src_rank == dst_rank or int(byte_count) <= 0:
                    continue
                flows.append(
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
        return tuple(flows)

    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(matrix0, "p0_dispatch", "ready", True),
            blocked_flows=_flows(matrix1, "p1_return", "blocked", False),
            forecast_pressure=_flows(matrix2, "p2_next_dispatch_forecast", "advisory_only", False),
        ),
        topology=LogicalTopology(num_gpus=int(payload["num_gpus"])),
        release_model=ReleaseConstraint(phase="p1_return", rank=0, release_after_phase="p0_dispatch", expert_compute_delay=0.0),
        forecast=ForecastPressure(
            source=p2_source,
            digest=name,
            oracle=p2_source == "perfect_trace",
            evaluation_eligible=p2_source not in {"perfect_trace", "shuffled_hint"},
            matrix_shape=(len(matrix2), len(matrix2[0]) if matrix2 else 0),
            matrix_total_bytes=sum(sum(row) for row in matrix2),
            matrix=matrix2,
        ),
        options=GlobalReadySetOptions(
            scheduling_mode="runtime_lookahead",
            information_mode="p0_p1_p2",
            prediction_confidence=1.0,
            p0_weight=1.0,
            p1_reservation_weight=1.0,
            p2_hint_weight=1.0,
        ),
        p0_dispatch_matrix=matrix0,
        p1_return_matrix=matrix1,
        p2_next_dispatch_forecast_matrix=matrix2,
    )


def _coverage(plan) -> dict[tuple[str, int, int], int]:
    covered = {}
    for wave in plan.waves:
        used_src = set()
        used_dst = set()
        for flow in wave.flows:
            assert flow.src_rank not in used_src
            assert flow.dst_rank not in used_dst
            used_src.add(flow.src_rank)
            used_dst.add(flow.dst_rank)
            key = (flow.phase, flow.src_rank, flow.dst_rank)
            covered[key] = covered.get(key, 0) + int(flow.byte_count)
    return covered


def test_phase_local_policies_cover_exactly_once() -> None:
    problem = _load_problem("receiver_incast_4rank")
    for policy_name in (
        "phase_barrier_fifo",
        "greedy_ready_set",
        "birkhoff_phase_local",
        "aurora_order_fixed",
        "fast_bvn_single_tier",
    ):
        plan = resolve_policy(policy_name=policy_name, bucket_rows=16).build_logical_plan(problem)
        covered = _coverage(plan)
        expected = {}
        for flow in problem.flow_window.ready_flows:
            expected[(flow.phase, flow.src_rank, flow.dst_rank)] = expected.get((flow.phase, flow.src_rank, flow.dst_rank), 0) + int(flow.byte_count)
        for flow in problem.flow_window.blocked_flows:
            expected[(flow.phase, flow.src_rank, flow.dst_rank)] = expected.get((flow.phase, flow.src_rank, flow.dst_rank), 0) + int(flow.byte_count)
        assert covered == expected


def test_phase_local_policies_schedule_real_p2_under_execution_window() -> None:
    problem = _load_problem("receiver_incast_4rank")
    problem = MultiPhaseSchedulingProblem(
        flow_window=problem.flow_window,
        topology=problem.topology,
        release_model=problem.release_model,
        forecast=problem.forecast,
        options=GlobalReadySetOptions(
            scheduling_mode="execution_window",
            information_mode="p0_p1_p2",
            prediction_confidence=1.0,
            p0_weight=1.0,
            p1_reservation_weight=1.0,
            p2_hint_weight=1.0,
        ),
        p0_dispatch_matrix=problem.p0_dispatch_matrix,
        p1_return_matrix=problem.p1_return_matrix,
        p2_next_dispatch_forecast_matrix=problem.p2_next_dispatch_forecast_matrix,
    )
    for policy_name in (
        "phase_barrier_fifo",
        "greedy_ready_set",
        "birkhoff_phase_local",
        "aurora_order_fixed",
        "fast_bvn_single_tier",
    ):
        plan = resolve_policy(policy_name=policy_name, bucket_rows=16).build_logical_plan(problem)
        assert plan.diagnostics["future_information_mode"] == "oracle_execution_window"
        assert plan.diagnostics["prediction_used"] is False
        assert plan.diagnostics["forecast_consumed"] is False
        assert plan.diagnostics["evaluation_eligible"] is False
        assert plan.diagnostics["valid"] is True, plan.diagnostics["audit"].get("validation_errors")
        assert any(flow.phase == "p2_next_dispatch" for wave in plan.waves for flow in wave.flows)


def test_phase_local_policies_do_not_consume_unused_oracle_forecast() -> None:
    problem = _load_problem("receiver_incast_4rank", p2_source="perfect_trace")
    for policy_name in (
        "phase_barrier_fifo",
        "greedy_ready_set",
        "birkhoff_phase_local",
        "aurora_order_fixed",
        "fast_bvn_single_tier",
    ):
        plan = resolve_policy(policy_name=policy_name, bucket_rows=16).build_logical_plan(problem)
        assert plan.diagnostics["future_information_mode"] == "none"
        assert plan.diagnostics["forecast_available"] is True
        assert plan.diagnostics["forecast_source"] == "perfect_trace"
        assert plan.diagnostics["forecast_consumed"] is False
        assert plan.diagnostics["prediction_used"] is False
        assert plan.diagnostics["evaluation_eligible"] is True
        assert all(flow.phase != "p2_next_dispatch" for wave in plan.waves for flow in wave.flows)


def test_offline_schedule_tail_metrics_capture_unlock_and_tail_signals() -> None:
    problem = _load_problem("receiver_incast_4rank")
    problem = MultiPhaseSchedulingProblem(
        flow_window=problem.flow_window,
        topology=problem.topology,
        release_model=ReleaseConstraint(phase="p1_return", rank=0, release_after_phase="p0_dispatch", expert_compute_delay=2.0),
        forecast=problem.forecast,
        options=GlobalReadySetOptions(
            scheduling_mode="execution_window",
            information_mode="p0_p1_p2",
            prediction_confidence=1.0,
            p0_weight=1.0,
            p1_reservation_weight=1.0,
            p2_hint_weight=1.0,
        ),
        p0_dispatch_matrix=problem.p0_dispatch_matrix,
        p1_return_matrix=problem.p1_return_matrix,
        p2_next_dispatch_forecast_matrix=problem.p2_next_dispatch_forecast_matrix,
    )
    plan = resolve_policy(policy_name="birkhoff_phase_local", bucket_rows=16).build_logical_plan(problem)
    audit = replay_and_audit_logical_plan(problem, plan)
    metrics = summarize_schedule_tail_metrics(problem=problem, plan=plan, audit=audit)
    assert metrics["active_wave_count"] > 0
    assert metrics["wave_duration_p95"] >= metrics["wave_duration_p50"]
    assert metrics["wave_duration_p99"] >= metrics["wave_duration_p95"]
    assert metrics["p0_inbound_completion_p95"] >= metrics["p0_inbound_completion_p50"]
    assert metrics["first_p1_release_time"] is not None
    assert metrics["first_p1_start_time"] is not None
    assert metrics["first_p1_start_time"] >= metrics["first_p1_release_time"]
    assert metrics["mean_p1_release_wait"] is not None
    assert metrics["max_p1_release_wait"] is not None
    assert metrics["bottleneck_send_busy_share"] is not None
    assert metrics["bottleneck_recv_busy_share"] is not None


def test_offline_schedule_tail_metrics_suppress_p2_release_in_runtime_lookahead() -> None:
    problem = _load_problem("receiver_incast_4rank", p2_source="perfect_trace")
    plan = resolve_policy(policy_name="greedy_ready_set", bucket_rows=16).build_logical_plan(problem)
    audit = replay_and_audit_logical_plan(problem, plan)
    metrics = summarize_schedule_tail_metrics(problem=problem, plan=plan, audit=audit)
    assert metrics["first_p2_release_time"] is None
    assert metrics["first_p2_start_time"] is None
    assert metrics["mean_p2_release_wait"] is None


def test_fast_bvn_respects_scale_limit() -> None:
    problem = _load_problem("skewed_8rank")
    plan = resolve_policy(policy_name="fast_bvn_single_tier", bucket_rows=16).build_logical_plan(problem)
    assert plan.diagnostics["wave_count"] > 0


def test_routersense_information_modes_diverge_on_sensitive_fixture() -> None:
    problem = _load_problem("p2_lookahead_sensitive_4rank")
    p0_only = resolve_policy(policy_name="routersense_multiphase_lookahead:p0_only", bucket_rows=16).build_logical_plan(problem)
    p0_p1 = resolve_policy(policy_name="routersense_multiphase_lookahead:p0_p1", bucket_rows=16).build_logical_plan(problem)
    p0_p1_p2 = resolve_policy(policy_name="routersense_multiphase_lookahead:p0_p1_p2", bucket_rows=16).build_logical_plan(problem)
    signatures = [tuple((flow.phase, flow.src_rank, flow.dst_rank, flow.byte_count) for wave in plan.waves for flow in wave.flows) for plan in (p0_only, p0_p1, p0_p1_p2)]
    assert len(set(signatures)) >= 2


def test_routersense_forecast_is_advisory_only() -> None:
    problem = _load_problem("p2_lookahead_sensitive_4rank")
    plan = resolve_policy(policy_name="routersense_multiphase_lookahead:p0_p1_p2", bucket_rows=16).build_logical_plan(problem)
    assert all(flow.phase != "p2_next_dispatch_forecast" for wave in plan.waves for flow in wave.flows)
    assert plan.diagnostics["p2_forecast_used"] is True


def test_native_passthrough_has_no_logical_plan() -> None:
    problem = _load_problem("phase_barrier_4rank")
    policy = resolve_policy(policy_name="native_passthrough", bucket_rows=16)
    try:
        policy.build_logical_plan(problem)
    except ValueError as exc:
        assert "does not build a logical schedule plan" in str(exc)
    else:
        raise AssertionError("native_passthrough should fail closed offline")
