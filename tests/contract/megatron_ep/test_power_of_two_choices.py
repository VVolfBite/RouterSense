from __future__ import annotations

from rs.scheduling.registry import resolve_phase_policy
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix


def test_power_of_two_choices_builds_valid_deterministic_plan() -> None:
    contexts = make_contexts_from_matrix(phase="P0", matrix=((0, 9, 3, 1), (7, 0, 2, 0), (5, 1, 0, 4), (2, 8, 6, 0)))
    policy = resolve_phase_policy(policy_name="power_of_two_choices", bucket_rows=4)
    plan_a = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
    plan_b = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
    assert plan_a.plan_hash == plan_b.plan_hash
    assert sum(len(wave.bucket_tasks) for wave in plan_a.waves) > 0
    assert plan_a.metrics["policy_name"] == "power_of_two_choices"
    assert plan_a.metrics["uses_p2"] is False
