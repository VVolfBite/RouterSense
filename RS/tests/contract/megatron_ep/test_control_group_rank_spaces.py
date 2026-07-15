from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from rs.runtime.online.megatron_ep.control import agreement_wire as agreement_wire_mod
from rs.runtime.online.megatron_ep.control import plan_agreement as plan_agreement_mod
from rs.runtime.online.megatron_ep.host import get_process_group_ranks_safe
from rs.scheduling.registry import resolve_phase_policy
from tests.contract.megatron_ep.helpers import make_observation, make_phase_context_generic


def test_get_process_group_ranks_safe_requires_explicit_rank_api(monkeypatch) -> None:
    monkeypatch.delattr("rs.runtime.online.megatron_ep.host.dist.get_process_group_ranks", raising=False)
    monkeypatch.setattr("rs.runtime.online.megatron_ep.host.dist.is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="explicit process-group rank order"):
        get_process_group_ranks_safe(object())


def test_get_process_group_ranks_safe_rejects_implicit_world_rank_guess(monkeypatch) -> None:
    monkeypatch.setattr("rs.runtime.online.megatron_ep.host.dist.is_initialized", lambda: True)
    monkeypatch.setattr("rs.runtime.online.megatron_ep.host.dist.get_world_size", lambda: 4)
    with pytest.raises(RuntimeError, match="explicit process-group rank order is required when no process group is provided"):
        get_process_group_ranks_safe(None)


def test_agreement_wire_rejects_missing_explicit_group_ranks(monkeypatch) -> None:
    monkeypatch.setattr(agreement_wire_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(agreement_wire_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(agreement_wire_mod.dist, "get_world_size", lambda: 4)
    with pytest.raises(RuntimeError, match="explicit_group_ranks are required"):
        agreement_wire_mod._resolve_group_rank_order(process_group=None, explicit_group_ranks=None)


def test_run_policy_agreement_uses_group_local_root_and_explicit_group(monkeypatch) -> None:
    process_group = object()
    local_observation = make_observation(rank=2, phase="P0", rows=(4, 0), ep_group_ranks=(2, 3), local_rank=0)
    context = SimpleNamespace(ep_group_ranks=(2, 3))

    class _Policy:
        def __init__(self) -> None:
            self.calls = 0

        def build_plan(self, *, local_context, global_contexts):
            self.calls += 1
            return SimpleNamespace(
                plan_hash="plan-hash",
                policy_name="joint_shadow_p0p1",
                policy_version="v1",
                control_mode="default_continue",
                observation_digest="obs-digest",
            )

    policy = _Policy()
    gather_calls: list[tuple[tuple[int, ...], object, int]] = []
    observation_payload_rank2 = [2] + [0] * 14 + [123]
    observation_payload_rank3 = [3] + [0] * 14 + [456]

    def _fake_all_gather(local_values, *, device, process_group=None, group_world_size=None):
        gather_calls.append((tuple(int(v) for v in local_values), process_group, int(group_world_size)))
        if local_values == [111]:
            return [observation_payload_rank2, observation_payload_rank3]
        if local_values == [99]:
            return [[99], [0]]
        return [[agreement_wire_mod._hash_to_i64("plan-hash")], [agreement_wire_mod._hash_to_i64("plan-hash")]]

    monkeypatch.setattr(agreement_wire_mod, "encode_runtime_observation", lambda observation: [111])
    monkeypatch.setattr(agreement_wire_mod, "_all_gather_variable_int64", _fake_all_gather)
    monkeypatch.setattr(
        agreement_wire_mod,
        "decode_runtime_observation",
        lambda payload, **kwargs: SimpleNamespace(global_rank=int(payload[0])),
    )
    monkeypatch.setattr(agreement_wire_mod, "encode_plan_tensor", lambda plan, ep_group_size: [99])
    monkeypatch.setattr(
        agreement_wire_mod,
        "decode_plan_tensor",
        lambda payload, ep_group_size=None: SimpleNamespace(
            plan_hash="plan-hash",
            policy_name="joint_shadow_p0p1",
            policy_version="v1",
            control_mode="default_continue",
            observation_digest="obs-digest",
        ),
    )
    monkeypatch.setattr(agreement_wire_mod.dist, "get_process_group_ranks", lambda group=None: [2, 3])

    plan, agreement = agreement_wire_mod.run_policy_agreement(
        local_observation=local_observation,
        policy=policy,
        context=context,
        group=process_group,
        device=torch.device("cpu"),
    )

    assert policy.calls == 1
    assert plan.plan_hash == "plan-hash"
    assert agreement.root_rank == 2
    assert [call[1] for call in gather_calls] == [process_group, process_group, process_group]
    assert [call[2] for call in gather_calls] == [2, 2, 2]


def test_run_phase_plan_agreement_uses_group_local_root_for_noncontiguous_group(monkeypatch) -> None:
    local_context = make_phase_context_generic(
        rank=2,
        phase="P0",
        input_splits=(4, 0),
        output_splits=(4, 0),
        ep_group_ranks=(2, 3),
    )
    remote_context = make_phase_context_generic(
        rank=3,
        phase="P0",
        input_splits=(0, 4),
        output_splits=(0, 4),
        ep_group_ranks=(2, 3),
    )
    policy = resolve_phase_policy(policy_name="phase_barrier_fifo", bucket_rows=0)
    encoded = [
        plan_agreement_mod._encode_planning_summary_tensor(
            ctx.to_planning_summary(),
            world_size=2,
            device=torch.device("cpu"),
        )
        for ctx in (local_context, remote_context)
    ]
    broadcast_state: dict[str, object] = {}

    monkeypatch.setattr(plan_agreement_mod.dist, "get_process_group_ranks", lambda group=None: [2, 3])
    monkeypatch.setattr(plan_agreement_mod.dist, "get_backend", lambda group=None: "gloo")

    def _all_gather(output, value, group=None):
        output[0].copy_(encoded[0])
        output[1].copy_(encoded[1])

    def _broadcast(tensor, src=0, group=None):
        broadcast_state.setdefault("srcs", []).append(int(src))
        if tensor.numel() == 1:
            payload = plan_agreement_mod._encode_abstract_plan_tensor(
                original_build(local_context=local_context, global_contexts=(local_context, remote_context)).to_abstract_plan(),
                device=torch.device("cpu"),
            )
            broadcast_state["payload"] = payload
            tensor.fill_(int(payload.numel()))
            return None
        tensor.copy_(broadcast_state["payload"])
        return None

    monkeypatch.setattr(plan_agreement_mod.dist, "all_gather", _all_gather)
    monkeypatch.setattr(plan_agreement_mod.dist, "broadcast", _broadcast)

    calls = {"count": 0}
    original_build = policy.build_plan

    def _counting_build(*, local_context, global_contexts):
        calls["count"] += 1
        return original_build(local_context=local_context, global_contexts=global_contexts)

    monkeypatch.setattr(policy, "build_plan", _counting_build)
    plan = plan_agreement_mod.run_phase_plan_agreement(
        local_context=local_context,
        policy=policy,
        group=object(),
    )

    assert calls["count"] == 1
    assert plan.plan_hash
    assert broadcast_state["srcs"] == [2, 2]
