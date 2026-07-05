from __future__ import annotations

from rs.scheduling.policy.registry import resolve_phase_policy
from .helpers import make_contexts_from_matrix


def test_aurora_order_fixed_emits_diagnostics() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 0, 0, 48), (0, 0, 0, 64), (0, 0, 0, 32), (16, 0, 0, 0)),
    )
    plan = resolve_phase_policy(policy_name="aurora_order_fixed", bucket_rows=16).build_plan(
        local_context=contexts[0],
        global_contexts=contexts,
    )
    diagnostics = plan.metrics["policy_diagnostics"]
    assert diagnostics["policy_name"] == "aurora_order_fixed"
    assert diagnostics["uses_current_phase_demand"] is True
    assert "selection_trace" in diagnostics["priority_components"]
    assert plan.metrics["policy_capabilities"]["uses_p2"] is False
