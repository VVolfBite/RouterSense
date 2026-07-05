from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs.core.experiment_config import SUPPORTED_ONLINE_PHASE_POLICIES
from rs.scheduling import GlobalReadySetOptions, LogicalTopology, MultiPhaseSchedulingProblem, ReleaseConstraint, resolve_phase_policy, resolve_policy
from rs.scheduling.contracts import FlowDemand, FlowWindow, ForecastPressure
from rs.scheduling.phase_local.islip_round_robin import ISLIPNoProgressError, ISLIPRoundRobinPolicy, _schedule_flows
from rs.scheduling.reference.birkhoff_von_neumann_fluid import decompose_fluid_matrix
from rs.scheduling.reference.exact_small_instance import MAX_BUCKET_TASK_COUNT, solve_exact_small_instance, solve_problem_exact
from rs.scheduling.validation import compare_plan_to_exact_reference, stable_hash, validate_bvn_fluid_certificate, validate_logical_plan


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "scheduling"


def _flows(matrix, phase, release_state, executable):
    flows = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank != dst_rank and int(byte_count) > 0:
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


def _problem(name: str) -> MultiPhaseSchedulingProblem:
    payload = json.loads((FIXTURE_ROOT / f"{name}.json").read_text(encoding="utf-8"))
    p0 = tuple(tuple(int(v) for v in row) for row in payload["p0_dispatch_matrix"])
    p1 = tuple(tuple(int(v) for v in row) for row in payload["p1_return_matrix"])
    p2 = tuple(tuple(int(v) for v in row) for row in payload["p2_next_dispatch_forecast_matrix"])
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(p0, "p0_dispatch", "ready", True),
            blocked_flows=_flows(p1, "p1_return", "blocked", False),
            forecast_pressure=_flows(p2, "p2_next_dispatch_forecast", "advisory_only", False),
        ),
        topology=LogicalTopology(num_gpus=int(payload["num_gpus"])),
        release_model=ReleaseConstraint(phase="p1_return", rank=0, release_after_phase="p0_dispatch"),
        forecast=ForecastPressure(
            source="copy_current_dispatch",
            digest=name,
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(len(p2), len(p2[0]) if p2 else 0),
            matrix_total_bytes=sum(sum(row) for row in p2),
            matrix=p2,
        ),
        options=GlobalReadySetOptions(scheduling_mode="runtime_lookahead"),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def _expected_flows(problem: MultiPhaseSchedulingProblem):
    return tuple(problem.flow_window.ready_flows + problem.flow_window.blocked_flows)


@pytest.mark.parametrize("fixture", ["phase_barrier_4rank", "receiver_incast_4rank", "full_duplex_pair_4rank", "skewed_8rank"])
def test_islip_round_robin_coverage_and_determinism(fixture: str) -> None:
    problem = _problem(fixture)
    policy = resolve_policy(policy_name="islip_round_robin", bucket_rows=16)
    plans = [policy.build_logical_plan(problem) for _ in range(3)]
    hashes = [stable_hash(plan.to_dict()) for plan in plans]
    assert hashes[0] == hashes[1] == hashes[2]
    validation = validate_logical_plan(plans[0], expected_flows=_expected_flows(problem))
    assert validation["valid"], validation["errors"]
    assert plans[0].diagnostics["islip_trace"]


def test_islip_pointer_seed_changes_legal_schedule() -> None:
    problem = _problem("pointer_sensitive_4rank")
    left = ISLIPRoundRobinPolicy(bucket_rows=16, pointer_seed="left").build_logical_plan(problem)
    right = ISLIPRoundRobinPolicy(bucket_rows=16, pointer_seed="right").build_logical_plan(problem)
    assert validate_logical_plan(left, expected_flows=_expected_flows(problem))["valid"]
    assert validate_logical_plan(right, expected_flows=_expected_flows(problem))["valid"]
    assert stable_hash(left.to_dict()) != stable_hash(right.to_dict())


def test_islip_fallback_and_no_progress_fail_closed() -> None:
    flows = (FlowDemand("f0", "p0_dispatch", 0, 1, 10, "ready", True),)
    waves, trace = _schedule_flows(flows, ranks=(2, 3), seed_payload={"case": "fallback"}, start_wave_id=0, max_rounds=1)
    assert waves
    assert trace[0]["fallback_used"] is True
    with pytest.raises(ISLIPNoProgressError):
        _schedule_flows(flows, ranks=(), seed_payload={"case": "no-progress"}, start_wave_id=0, max_rounds=1)


@pytest.mark.parametrize("matrix", [
    ((0, 10, 0, 0), (0, 0, 3, 0), (4, 0, 0, 8), (1, 0, 0, 0)),
    ((0, 9, 0, 0), (0, 0, 0, 0), (0, 8, 0, 0), (0, 7, 0, 0)),
    ((0, 5, 0, 0), (5, 0, 0, 0), (0, 0, 0, 6), (0, 0, 6, 0)),
])
def test_bvn_fluid_certificate(matrix) -> None:
    waves, certificate = decompose_fluid_matrix(matrix, phase="p0_dispatch")
    cert = certificate.to_dict()
    validation = validate_bvn_fluid_certificate(cert)
    assert validation["valid"], validation["errors"]
    row_load = max(sum(int(v) for dst, v in enumerate(row) if src != dst) for src, row in enumerate(matrix))
    col_load = max(sum(int(matrix[src][dst]) for src in range(len(matrix)) if src != dst) for dst in range(len(matrix)))
    assert cert["fluid_optimal_horizon"] == max(row_load, col_load)
    assert all("flow_id" not in str(wave.get("dummy_edges", "")) for wave in cert["waves"])
    assert validate_logical_plan(type("P", (), {"waves": tuple(waves)})(), expected_flows=_flows(matrix, "p0_dispatch", "ready", True))["valid"]


def test_exact_small_instance_optimum_and_scale_limit() -> None:
    flows = (
        FlowDemand("a", "p0_dispatch", 0, 1, 10, "ready", True),
        FlowDemand("b", "p0_dispatch", 1, 0, 8, "ready", True),
        FlowDemand("c", "p0_dispatch", 0, 2, 3, "ready", True),
    )
    result = solve_exact_small_instance(flows=flows, rank_count=3, time_limit_ms=5000)
    assert result["solver_status"] == "optimal"
    assert result["certified_optimal"] is True
    assert result["objective_logical_makespan"] == 13
    too_many = tuple(
        FlowDemand(f"f{i}", "p0_dispatch", i % 4, (i + 1) % 4, 1, "ready", True)
        for i in range(MAX_BUCKET_TASK_COUNT + 1)
    )
    unsupported = solve_exact_small_instance(flows=too_many, rank_count=4)
    assert unsupported["supported"] is False
    assert unsupported["solver_status"] == "unsupported_scale"


def test_exact_reference_detects_nonoptimal_heuristic_and_optimal_policy() -> None:
    problem = _problem("exact_nonoptimal_4rank")
    exact = solve_problem_exact(problem)
    assert exact["certified_optimal"] is True
    greedy = resolve_policy(policy_name="greedy_ready_set", bucket_rows=16).build_logical_plan(problem)
    exact_policy = resolve_policy(policy_name="exact_small_instance_reference", bucket_rows=16).build_logical_plan(problem)
    greedy_cmp = compare_plan_to_exact_reference(greedy, exact)
    exact_policy_cmp = compare_plan_to_exact_reference(exact_policy, exact)
    assert greedy_cmp["optimality_gap"] > 0
    assert exact_policy_cmp["policy_reaches_optimum"] is True


def test_policy_matrix_consistency() -> None:
    assert "islip_round_robin" in SUPPORTED_ONLINE_PHASE_POLICIES
    assert resolve_policy(policy_name="islip_round_robin", bucket_rows=16).capabilities.supports_online_phase_local_execution
    assert not resolve_policy(policy_name="birkhoff_von_neumann_fluid", bucket_rows=16).capabilities.supports_online_phase_local_execution
    assert not resolve_policy(policy_name="exact_small_instance_reference", bucket_rows=16).capabilities.supports_online_phase_local_execution
    with pytest.raises(Exception):
        resolve_phase_policy(policy_name="birkhoff_von_neumann_fluid", bucket_rows=16)
    with pytest.raises(Exception):
        resolve_phase_policy(policy_name="exact_small_instance_reference", bucket_rows=16)
    with pytest.raises(Exception):
        resolve_phase_policy(policy_name="routersense_multiphase_lookahead:p0_p1_p2", bucket_rows=16)
