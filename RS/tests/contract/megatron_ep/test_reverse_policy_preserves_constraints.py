from __future__ import annotations

from integrations.megatron_ep.routersense.policy.registry import resolve_phase_policy
from .helpers import make_phase_context


def test_reverse_policy_preserves_full_duplex_constraints() -> None:
    ctx0 = make_phase_context(rank=0, phase="P0", input_splits=(0, 48), output_splits=(0, 48), rows=48)
    ctx1 = make_phase_context(rank=1, phase="P0", input_splits=(48, 0), output_splits=(48, 0), rows=48)
    plan = resolve_phase_policy(policy_name="trivial_reverse_bucket", bucket_rows=16).build_plan(
        local_context=ctx0,
        global_contexts=(ctx0, ctx1),
    )
    for wave in plan.waves:
        outgoing = set()
        incoming = set()
        for task in wave.bucket_tasks:
            assert task.src_rank not in outgoing
            assert task.dst_rank not in incoming
            outgoing.add(task.src_rank)
            incoming.add(task.dst_rank)


def test_reverse_policy_preserves_p0_bundle_atomicity() -> None:
    ctx0 = make_phase_context(rank=0, phase="P0", input_splits=(0, 48), output_splits=(0, 48), rows=48)
    ctx1 = make_phase_context(rank=1, phase="P0", input_splits=(48, 0), output_splits=(48, 0), rows=48)
    plan = resolve_phase_policy(policy_name="trivial_reverse_bucket", bucket_rows=16).build_plan(
        local_context=ctx0,
        global_contexts=(ctx0, ctx1),
    )
    for wave in plan.waves:
        for task in wave.bucket_tasks:
            roles = [payload.tensor_role for payload in task.payload_slices]
            assert roles == ["hidden_states", "routing_probs"]


def test_reverse_policy_ignores_p2_hint() -> None:
    ctx0_none = make_phase_context(rank=0, phase="P0", input_splits=(0, 48), output_splits=(0, 48), rows=48, p2_hint_mode="none")
    ctx1_none = make_phase_context(rank=1, phase="P0", input_splits=(48, 0), output_splits=(48, 0), rows=48, p2_hint_mode="none")
    ctx0_stub = make_phase_context(rank=0, phase="P0", input_splits=(0, 48), output_splits=(0, 48), rows=48, p2_hint_mode="deterministic_stub")
    ctx1_stub = make_phase_context(rank=1, phase="P0", input_splits=(48, 0), output_splits=(48, 0), rows=48, p2_hint_mode="deterministic_stub")
    policy = resolve_phase_policy(policy_name="trivial_reverse_bucket", bucket_rows=16)
    none_plan = policy.build_plan(local_context=ctx0_none, global_contexts=(ctx0_none, ctx1_none))
    stub_plan = policy.build_plan(local_context=ctx0_stub, global_contexts=(ctx0_stub, ctx1_stub))
    none_order = [task.task_id for wave in none_plan.waves for task in wave.bucket_tasks]
    stub_order = [task.task_id for wave in stub_plan.waves for task in wave.bucket_tasks]
    assert none_order == stub_order
    assert none_plan.metrics["p2_influenced_plan"] is False
    assert stub_plan.metrics["p2_influenced_plan"] is False
