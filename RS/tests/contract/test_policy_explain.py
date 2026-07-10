from __future__ import annotations

import json
from pathlib import Path

from experiments.offline.replay_fixture_policy_study import _build_problem
from rs.scheduling.policy_explain import explain_policy_decision
from rs.scheduling.registry import resolve_policy


def _write_fixture(path: Path) -> dict:
    fixture = {
        "num_gpus": 2,
        "p0_dispatch_matrix": [[0, 8], [4, 0]],
        "p1_return_matrix": [[0, 4], [8, 0]],
        "p2_next_dispatch_forecast_matrix": [[0, 10], [5, 0]],
        "p2_next_dispatch_matrix": [[0, 10], [5, 0]],
        "metadata": {"layer_id": "0", "next_layer_id": "1"},
    }
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return fixture


def test_policy_explain_is_deterministic(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "replay_layer_0.json")
    problem = _build_problem(fixture, mode="runtime_lookahead", p2_source="actual_trace", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0)
    first = explain_policy_decision(policy, problem, p2_hint=problem.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    second = explain_policy_decision(policy, problem, p2_hint=problem.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    assert first.to_dict() == second.to_dict()


def test_explain_selected_order_matches_real_policy_order(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "replay_layer_0.json")
    problem = _build_problem(fixture, mode="runtime_lookahead", p2_source="actual_trace", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="RS_safe_gated_greedy", bucket_rows=0)
    explain = explain_policy_decision(policy, problem, p2_hint=problem.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    plan = policy.build_logical_plan(problem)
    actual_order = tuple(flow.flow_id for wave in plan.waves for flow in wave.flows)
    assert explain.selected_order == actual_order


def test_p2_zero_has_zero_p2_contribution(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "replay_layer_0.json")
    problem = _build_problem(fixture, mode="runtime_lookahead", p2_source="zero_hint", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0)
    zero_matrix = tuple(tuple(0 for _ in row) for row in problem.p2_next_dispatch_forecast_matrix)
    explain = explain_policy_decision(policy, problem, p2_hint=zero_matrix, p2_source="zero_hint")
    assert all(value == 0.0 for _edge, value in explain.p2_score_by_edge)
