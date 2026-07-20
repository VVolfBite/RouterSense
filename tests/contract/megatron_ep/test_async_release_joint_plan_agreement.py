from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    GlobalJointPlanWire,
    agree_global_joint_plan,
    validate_local_schedule_against_global_plan,
    validate_pairwise_send_recv_contracts,
)


def _wire() -> GlobalJointPlanWire:
    return GlobalJointPlanWire(
        window_key="run:epoch1:mb0:layer5",
        policy_name="routersense_joint_zero_hint_async_p2p",
        safe_selected_policy="future:p012:joint:global:rscf",
        prediction_digest="pred0",
        canonical_edge_order=(("P0", 2, 3), ("P1", 3, 2)),
        wave_metadata=((0, (("P0", 2, 3),)), (1, (("P1", 3, 2),))),
        per_peer_sequence_digest="seq0",
    )


def test_global_plan_same_local_length_can_differ() -> None:
    wire = _wire()
    result = agree_global_joint_plan(wire, gathered_wires=(wire, wire))
    assert result["valid"] is True
    rank0_schedule = (
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 2, "payload_roles": ("hidden_states", "routing_probs"), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 3, "payload_roles": ("hidden_states", "routing_probs"), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P1", "src_rank": 3, "dst_rank": 2, "owner_global_rank": 2, "payload_roles": ("hidden_states",), "dtype": "torch.float16", "row_count": 4},
    )
    rank1_schedule = (
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 2, "payload_roles": ("hidden_states",), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 2, "payload_roles": ("routing_probs",), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 3, "payload_roles": ("hidden_states",), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 3, "payload_roles": ("routing_probs",), "dtype": "torch.float16", "row_count": 4},
        {"phase": "P1", "src_rank": 3, "dst_rank": 2, "owner_global_rank": 2, "payload_roles": ("hidden_states",), "dtype": "torch.float16", "row_count": 4},
    )
    assert validate_local_schedule_against_global_plan(rank0_schedule, global_wire=wire)["valid"] is True
    assert validate_local_schedule_against_global_plan(rank1_schedule, global_wire=wire)["valid"] is True


def test_global_plan_digest_mismatch_fails_before_execution() -> None:
    first = _wire()
    second = GlobalJointPlanWire(
        window_key=first.window_key,
        policy_name=first.policy_name,
        safe_selected_policy=first.safe_selected_policy,
        prediction_digest="pred1",
        canonical_edge_order=first.canonical_edge_order,
        wave_metadata=first.wave_metadata,
        per_peer_sequence_digest=first.per_peer_sequence_digest,
    )
    result = agree_global_joint_plan(first, gathered_wires=(first, second))
    assert result["valid"] is False
    assert "global_plan_digest_mismatch" in result["errors"]


def test_pairwise_send_recv_contracts_validate_matching_roles_and_rows() -> None:
    schedules = (
        (
            {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 2, "payload_roles": ("hidden_states", "routing_probs"), "dtype": "torch.float16", "row_count": 4},
        ),
        (
            {"phase": "P0", "src_rank": 2, "dst_rank": 3, "owner_global_rank": 3, "payload_roles": ("hidden_states", "routing_probs"), "dtype": "torch.float16", "row_count": 4},
        ),
    )
    assert validate_pairwise_send_recv_contracts(schedules)["valid"] is True
