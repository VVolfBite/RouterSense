from __future__ import annotations

import pytest

from integrations.megatron_ep.routersense.policy.registry import resolve_phase_policy
from integrations.megatron_ep.tests.helpers import make_contexts_from_matrix, with_p2_digest


def test_routersense_p0p1p2_hint_rejects_missing_hint() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 32, 32, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        p2_hint_mode="none",
    )
    with pytest.raises(ValueError, match="missing_p2_hint"):
        resolve_phase_policy(policy_name="routersense_p0p1p2_hint", bucket_rows=16).build_plan(
            local_context=contexts[0],
            global_contexts=contexts,
        )


def test_routersense_p0p1p2_hint_uses_stub_but_marks_not_evaluation_eligible() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 32, 32, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        p2_hint_mode="deterministic_stub",
    )
    plan = resolve_phase_policy(policy_name="routersense_p0p1p2_hint", bucket_rows=16).build_plan(
        local_context=contexts[0],
        global_contexts=contexts,
    )
    diagnostics = plan.metrics["policy_diagnostics"]
    assert diagnostics["p2_hint_available"] is True
    assert diagnostics["evaluation_eligible"] is False


def test_routersense_p0p1p2_hint_changes_with_hint_digest() -> None:
    base_contexts = list(
        make_contexts_from_matrix(
            phase="P0",
            matrix=((0, 32, 32, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
            p2_hint_mode="deterministic_stub",
        )
    )
    policy = resolve_phase_policy(policy_name="routersense_p0p1p2_hint", bucket_rows=16)
    signatures: dict[str, list[str]] = {}
    for digest in ["aaa111", "bbb222", "ccc333", "ddd444", "eee555", "fff666", "012345", "deadbeef"]:
        contexts = tuple(with_p2_digest(ctx, digest=digest) for ctx in base_contexts)
        plan = policy.build_plan(local_context=contexts[0], global_contexts=contexts)
        signatures[digest] = [task.task_id for wave in plan.waves for task in wave.bucket_tasks]
    assert len({tuple(value) for value in signatures.values()}) >= 2
