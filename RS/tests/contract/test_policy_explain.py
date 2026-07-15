from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from rs.runtime.offline.policy_study import build_replay_problem
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
    problem = build_replay_problem(fixture, mode="runtime_lookahead", p2_source="actual_trace", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0)
    first = explain_policy_decision(policy, problem, p2_hint=problem.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    second = explain_policy_decision(policy, problem, p2_hint=problem.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    assert first.to_dict() == second.to_dict()


def test_explain_selected_order_matches_real_policy_order(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "replay_layer_0.json")
    problem = build_replay_problem(fixture, mode="runtime_lookahead", p2_source="actual_trace", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="RS_safe_gated_greedy", bucket_rows=0)
    explain = explain_policy_decision(policy, problem, p2_hint=problem.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    plan = policy.build_logical_plan(problem)
    actual_order = tuple(flow.flow_id for wave in plan.waves for flow in wave.flows)
    assert explain.selected_order == actual_order
    assert explain.safe_selected_order == actual_order


def test_p2_zero_has_zero_p2_contribution(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "replay_layer_0.json")
    problem = build_replay_problem(fixture, mode="runtime_lookahead", p2_source="zero_hint", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0)
    zero_matrix = tuple(tuple(0 for _ in row) for row in problem.p2_next_dispatch_forecast_matrix)
    explain = explain_policy_decision(policy, problem, p2_hint=zero_matrix, p2_source="zero_hint")
    assert all(value == 0.0 for _edge, value in explain.p2_score_by_edge)


def test_safe_explain_order_matches_paired_b_on_fallback(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "replay_layer_0.json")
    fixture["p2_next_dispatch_forecast_matrix"] = [[0, 100], [1, 0]]
    fixture["p2_next_dispatch_matrix"] = [[0, 100], [1, 0]]
    problem = build_replay_problem(fixture, mode="runtime_lookahead", p2_source="actual_trace", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="RS_safe_gated_greedy", bucket_rows=0)
    explain = explain_policy_decision(policy, problem, p2_hint=problem.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    if explain.fallback_to_b:
        assert explain.safe_selected_order == explain.paired_b_selected_order
        assert explain.selected_order == explain.paired_b_selected_order
    else:
        assert explain.safe_selected_order == explain.raw_u_selected_order


def test_top_score_edges_not_labeled_critical_path(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "replay_layer_0.json")
    problem = build_replay_problem(fixture, mode="runtime_lookahead", p2_source="actual_trace", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0)
    explain = explain_policy_decision(policy, problem, p2_hint=problem.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    assert explain.top_score_edges
    assert explain.critical_path_edges is None
    assert explain.critical_path_unavailable_reason


def test_prediction_confidence_applied_once(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "replay_layer_0.json")
    base = build_replay_problem(fixture, mode="runtime_lookahead", p2_source="actual_trace", expert_compute_delay=0.0)
    policy = resolve_policy(policy_name="U_barrier_criticality_global_matching", bucket_rows=0)
    explain_full = explain_policy_decision(policy, base, p2_hint=base.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    half = replace(base, options=replace(base.options, prediction_confidence=0.5))
    explain_half = explain_policy_decision(policy, half, p2_hint=half.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    quarter = replace(base, options=replace(base.options, prediction_confidence=0.25))
    explain_quarter = explain_policy_decision(policy, quarter, p2_hint=quarter.p2_next_dispatch_forecast_matrix, p2_source="actual_trace_oracle")
    full_norm = sum(value for _edge, value in explain_full.p2_score_by_edge)
    half_norm = sum(value for _edge, value in explain_half.p2_score_by_edge)
    quarter_norm = sum(value for _edge, value in explain_quarter.p2_score_by_edge)
    assert abs(half_norm - 0.5 * full_norm) < 1e-6
    assert abs(quarter_norm - 0.25 * full_norm) < 1e-6
