from __future__ import annotations

from rs.scheduling.unified_interface import PolicyOptions, build_policy, build_request_from_replay_window
from rs.scheduling.wire_cost import phase_wire_bytes


class _ReplayWindow:
    fixture_id = "fixture"
    window_id = "window"
    layer_id = 0
    group_size = 3
    p0_truth_rows = ((0, 2, 5), (3, 0, 3), (1, 5, 0))
    p1_truth_rows = ((0, 3, 1), (2, 0, 5), (5, 3, 0))
    p2_truth_rows = ((0, 1, 2), (3, 0, 1), (1, 4, 0))


def test_barrier_common_core_metadata_matches_between_b_and_u() -> None:
    request = build_request_from_replay_window(
        replay_window=_ReplayWindow(),
        p2_hint_rows=_ReplayWindow.p2_truth_rows,
        hint_type="copy_current_dispatch",
        confidence=1.0,
        bucket_rows=0,
        policy_options=PolicyOptions(),
    )
    b_plan = build_policy("barrier_criticality_core_independent", PolicyOptions()).plan(request)
    u_plan = build_policy("barrier_criticality_joint", PolicyOptions()).plan(request)
    b_core = dict((b_plan.diagnostics or {}).get("common_core", {}))
    u_core = dict((u_plan.diagnostics or {}).get("common_core", {}))
    assert b_core["matching_core_id"] == u_core["matching_core_id"]
    assert b_core["task_contract_digest"] == u_core["task_contract_digest"]
    assert b_core["bucket_contract_digest"] == u_core["bucket_contract_digest"]
    assert b_core["cost_contract_digest"] == u_core["cost_contract_digest"]
    assert b_core["service_model_id"] == u_core["service_model_id"]
    assert b_core["solver_budget_digest"] == u_core["solver_budget_digest"]
    assert b_core["phase_independent"] is True
    assert u_core["phase_independent"] is False


def test_phase_wire_bytes_distinguishes_p0_and_p1_payloads() -> None:
    p0_hidden = phase_wire_bytes(
        phase="P0",
        tensor_role="hidden_states",
        rows=4,
        hidden_size=8,
        hidden_dtype="fp16",
        routing_probability_dtype="fp32",
        top_k=2,
    )
    p0_bundle = phase_wire_bytes(
        phase="P0",
        tensor_role="dispatch_bundle",
        rows=4,
        hidden_size=8,
        hidden_dtype="fp16",
        routing_probability_dtype="fp32",
        top_k=2,
    )
    p1_hidden = phase_wire_bytes(
        phase="P1",
        tensor_role="hidden_states",
        rows=4,
        hidden_size=8,
        hidden_dtype="fp16",
        routing_probability_dtype="fp32",
        top_k=2,
    )
    assert p0_bundle > p0_hidden
    assert p0_bundle > p1_hidden
    assert p1_hidden == p0_hidden
