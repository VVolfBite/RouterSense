from __future__ import annotations

from rs.scheduling.policy.registry import resolve_phase_policy
from .helpers import make_contexts_from_matrix


def test_routersense_p0p1_reservation_sees_future_pressure() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 16, 16, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 64, 0, 0)),
    )
    plan = resolve_phase_policy(
        policy_name="routersense_p0p1_reservation",
        bucket_rows=16,
        p0_weight=1.0,
        p1_reservation_weight=1.0,
    ).build_plan(local_context=contexts[0], global_contexts=contexts)
    diagnostics = plan.metrics["policy_diagnostics"]
    assert diagnostics["p1_reservation_seen"] is True
    assert diagnostics["p1_reservation_influenced_p0_plan"] is True
    assert diagnostics["p1_reservation_bytes"] > 0
