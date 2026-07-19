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

        def build_plan(self, *, context, global_observation):
            self.calls += 1
            return SimpleNamespace(
                plan_hash="plan-hash",
                policy_name="joint_shadow_p0p1",
                policy_version="v1",
                control_mode="default_continue",
                observation_digest="obs-digest",
            )

    policy = _Policy()
    gather_calls: list[tuple[str, object, int]] = []

    def _fake_all_gather_objects(local_value, *, process_group=None, group_world_size=None):
        gather_calls.append((type(local_value).__name__, process_group, int(group_world_size)))
        if isinstance(local_value, dict) and "observations_by_phase" in local_value:
            remote = make_observation(rank=3, phase="P0", rows=(0, 4), ep_group_ranks=(2, 3), local_rank=1)
            return [local_value, agreement_wire_mod.ObservationBundle(
                run_id=remote.run_id,
                forward_generation=0,
                microbatch_id=remote.microbatch_id,
                layer_id=remote.layer_id,
                ep_group_ranks=remote.ep_group_ranks,
                observations_by_phase={"P0": remote},
            ).to_dict()]
        if isinstance(local_value, dict) and "success" in local_value:
            return [dict(local_value), dict(local_value)]
        return ["plan-hash", "plan-hash"]

    monkeypatch.setattr(agreement_wire_mod, "_all_gather_objects", _fake_all_gather_objects)
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
    monkeypatch.setattr(
        agreement_wire_mod.dist,
        "broadcast_object_list",
        lambda object_list, src, group=None: None,
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
    monkeypatch.setattr(plan_agreement_mod.dist, "broadcast_object_list", lambda object_list, src, group=None: None)

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


@pytest.mark.parametrize(
    "failure_mode",
    ("planner_raises", "candidate_missing", "decoder_failure"),
)
def test_run_policy_agreement_root_failure_broadcasts_shared_status(monkeypatch, failure_mode: str) -> None:
    process_group = object()
    local_observation = make_observation(rank=2, phase="P0", rows=(4, 0), ep_group_ranks=(2, 3), local_rank=0)
    context = SimpleNamespace(ep_group_ranks=(2, 3))
    remote = make_observation(rank=3, phase="P0", rows=(0, 4), ep_group_ranks=(2, 3), local_rank=1)

    class _Policy:
        def build_plan(self, *, context, global_observation):
            if failure_mode == "planner_raises":
                raise RuntimeError("planner boom")
            if failure_mode == "candidate_missing":
                return None
            return SimpleNamespace(
                plan_hash="plan-hash",
                policy_name="joint_shadow_p0p1",
                policy_version="v1",
                control_mode="default_continue",
                observation_digest="obs-digest",
            )

    def _fake_all_gather_objects(local_value, *, process_group=None, group_world_size=None):
        if isinstance(local_value, dict) and "observations_by_phase" in local_value:
            return [
                agreement_wire_mod.ObservationBundle(
                    run_id=local_observation.run_id,
                    forward_generation=0,
                    microbatch_id=local_observation.microbatch_id,
                    layer_id=local_observation.layer_id,
                    ep_group_ranks=local_observation.ep_group_ranks,
                    observations_by_phase={"P0": local_observation},
                ).to_dict(),
                agreement_wire_mod.ObservationBundle(
                    run_id=remote.run_id,
                    forward_generation=0,
                    microbatch_id=remote.microbatch_id,
                    layer_id=remote.layer_id,
                    ep_group_ranks=remote.ep_group_ranks,
                    observations_by_phase={"P0": remote},
                ).to_dict(),
            ]
        if isinstance(local_value, dict) and "success" in local_value:
            return [dict(local_value), dict(local_value)]
        return ["mismatch-left", "mismatch-right"] if failure_mode == "decoder_failure" else ["shared", "shared"]

    status_payloads: list[dict[str, object]] = []

    def _broadcast_object_list(object_list, src, group=None):
        if object_list and isinstance(object_list[0], dict):
            status_payloads.append(dict(object_list[0]))

    monkeypatch.setattr(agreement_wire_mod, "_all_gather_objects", _fake_all_gather_objects)
    monkeypatch.setattr(agreement_wire_mod.dist, "broadcast_object_list", _broadcast_object_list)
    monkeypatch.setattr(agreement_wire_mod.dist, "get_process_group_ranks", lambda group=None: [2, 3])
    monkeypatch.setattr(agreement_wire_mod, "encode_plan_tensor", lambda plan, ep_group_size: [99])
    monkeypatch.setattr(
        agreement_wire_mod,
        "decode_plan_tensor",
        lambda payload, ep_group_size=None: (_ for _ in ()).throw(RuntimeError("decode boom"))
        if failure_mode == "decoder_failure"
        else SimpleNamespace(
            plan_hash="plan-hash",
            policy_name="joint_shadow_p0p1",
            policy_version="v1",
            control_mode="default_continue",
            observation_digest="obs-digest",
        ),
    )

    with pytest.raises(RuntimeError, match="policy_agreement_failed"):
        agreement_wire_mod.run_policy_agreement(
            local_observation=local_observation,
            policy=_Policy(),
            context=context,
            group=process_group,
            device=torch.device("cpu"),
        )

    assert status_payloads
    if failure_mode == "decoder_failure":
        assert status_payloads[-1]["success"] is True
    else:
        assert status_payloads[-1]["success"] is False
    assert status_payloads[-1]["root_global_rank"] == 2


@pytest.mark.parametrize("failure_mode", ("planner_raises", "validation_false"))
def test_run_phase_plan_agreement_root_failure_broadcasts_shared_status(monkeypatch, failure_mode: str) -> None:
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
    shared_status: list[dict[str, object]] = []

    monkeypatch.setattr(plan_agreement_mod.dist, "get_process_group_ranks", lambda group=None: [2, 3])
    monkeypatch.setattr(plan_agreement_mod.dist, "get_backend", lambda group=None: "gloo")
    monkeypatch.setattr(
        plan_agreement_mod.dist,
        "broadcast_object_list",
        lambda object_list, src, group=None: shared_status.append(dict(object_list[0])) if object_list and isinstance(object_list[0], dict) else None,
    )

    def _all_gather(output, value, group=None):
        output[0].copy_(encoded[0])
        output[1].copy_(encoded[1])

    monkeypatch.setattr(plan_agreement_mod.dist, "all_gather", _all_gather)
    monkeypatch.setattr(plan_agreement_mod.dist, "broadcast", lambda tensor, src=0, group=None: None)

    original_build = policy.build_plan

    def _failing_build(*, local_context, global_contexts):
        if failure_mode == "planner_raises":
            raise RuntimeError("planner boom")
        plan = original_build(local_context=local_context, global_contexts=global_contexts)
        return SimpleNamespace(
            to_abstract_plan=lambda: SimpleNamespace(validate=lambda: (_ for _ in ()).throw(ValueError("invalid")))
        )

    monkeypatch.setattr(policy, "build_plan", _failing_build)

    with pytest.raises(RuntimeError, match="phase_plan_agreement_failed"):
        plan_agreement_mod.run_phase_plan_agreement(
            local_context=local_context,
            policy=policy,
            group=object(),
        )

    assert shared_status
    assert shared_status[-1]["success"] is False
    assert shared_status[-1]["root_global_rank"] == 2
