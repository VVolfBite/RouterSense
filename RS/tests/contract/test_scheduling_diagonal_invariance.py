from __future__ import annotations

from rs.runtime.offline.policy_study import build_replay_problem
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling import resolve_policy


POLICIES = (
    "B_birkhoff",
    "B_birkhoff_wave",
    "B_gated_maxweight_matching",
    "B_barrier_price_adaptive_matching",
    "U_gated_maxweight_matching",
    "U_barrier_criticality_global_matching",
    "U_barrier_price_adaptive_matching",
    "U_lagrangian",
    "U_ibbr",
)


def _fixture_with(diagonal: int) -> dict[str, object]:
    return {
        "num_gpus": 3,
        "p0_dispatch_matrix": [[diagonal, 16, 4], [8, diagonal, 12], [6, 10, diagonal]],
        "p1_return_matrix": [[diagonal, 7, 3], [5, diagonal, 11], [9, 13, diagonal]],
        "p2_next_dispatch_forecast_matrix": [[diagonal, 14, 2], [4, diagonal, 6], [8, 12, diagonal]],
        "p2_next_dispatch_matrix": [[diagonal, 14, 2], [4, diagonal, 6], [8, 12, diagonal]],
        "metadata": {"layer_id": "0", "next_layer_id": "1"},
    }


def _result(policy_name: str, fixture: dict[str, object]) -> tuple[float, int, int]:
    mode = "execution_window" if policy_name.startswith("B_") or policy_name.startswith("U_") else "runtime_lookahead"
    problem = build_replay_problem(
        fixture,
        mode=mode,
        p2_source="actual_trace" if mode == "execution_window" else "copy_current_dispatch",
        expert_compute_delay=0.0,
    )
    plan = resolve_policy(policy_name=policy_name, bucket_rows=0).build_logical_plan(problem)
    audit = replay_and_audit_logical_plan(problem, plan)
    return float(plan.diagnostics.get("makespan", audit.get("makespan", 0.0))), int(plan.diagnostics.get("logical_flow_count", 0)), int(sum(flow.byte_count for wave in plan.waves for flow in wave.flows))


def test_diagonal_self_bytes_do_not_change_core_schedule_metrics() -> None:
    clean = _fixture_with(0)
    dirty = _fixture_with(10_000)
    for policy_name in POLICIES:
        assert _result(policy_name, clean) == _result(policy_name, dirty), policy_name
