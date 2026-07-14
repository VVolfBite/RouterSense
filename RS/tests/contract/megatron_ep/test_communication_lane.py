from __future__ import annotations

from rs.core.contracts import WindowPlan
from rs.planning.api import to_logical_plan
from rs.runtime.online.megatron_ep.control.communication_lane import GlooControlCommunicationLane, slot_from_request
from rs.runtime.online.megatron_ep.public_types import (
    LocalPreparationToken,
    LocalPublicationCandidate,
    PublicationPollStatus,
)
from rs.runtime.online.megatron_ep.target_planning.contracts import TargetLayerPreparedJointPlan
import rs.runtime.online.megatron_ep.control.communication_lane as lane_mod
from rs.scheduling.validation import stable_hash


def _candidate(*, slot_digest: str, logical_plan_digest: str = "ld") -> LocalPublicationCandidate:
    slot = slot_from_request(
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
    )
    window_plan = WindowPlan(
        planner_id="u",
        planner_family="joint",
        request_digest="req",
        waves=(),
        metadata={"legacy_policy_name": "u"},
    )
    logical_plan = to_logical_plan(window_plan)
    effective_digest = window_plan.semantic_digest() if logical_plan_digest == "ld" else logical_plan_digest
    plan = TargetLayerPreparedJointPlan(
        source_layer_id="0",
        target_layer_id="1",
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        h1_prediction_digest="h1",
        h2_prediction_digest="h2",
        target_problem_digest="tp",
        window_plan=window_plan,
        logical_plan=logical_plan,
        logical_plan_digest=effective_digest,
        legacy_logical_plan_digest=stable_hash(logical_plan.to_dict()),
        policy="u",
        weights={},
        bucket_contract_digest="bucket",
        topology_digest="topo",
        h1_rows=((0, 1), (1, 0)),
        derived_p1_rows=((0, 1), (1, 0)),
        h2_rows=((0, 1), (1, 0)),
        created_at_ns=1,
        ready_at_ns=2,
    )
    return LocalPublicationCandidate(
        slot=slot,
        planner_id="u",
        logical_plan_digest=effective_digest,
        token=LocalPreparationToken(
            service_session_id=1,
            forward_generation=1,
            target_layer_id="1",
            task_version=1,
            publication_slot_digest=slot_digest,
        ),
        status="READY",
        metadata={"plan": plan.to_dict()},
    )


def test_lane_rejects_slot_digest_mismatch(monkeypatch) -> None:
    slot = slot_from_request(
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
    )
    lane = GlooControlCommunicationLane(rank=0, world_size=2, root_rank=0, process_group=None, group_ranks=(0, 2))
    local = _candidate(slot_digest=slot.semantic_digest())
    monkeypatch.setattr(
        lane,
        "_all_gather_status",
        lambda payload: [
            dict(payload),
            {
                "slot_digest": "different",
                "group_rank": 1,
                "global_rank": 2,
                "status": "READY",
                "candidate": local.to_dict(),
            },
        ],
    )
    result = lane.poll(slot, local)
    assert result.status is PublicationPollStatus.SLOT_MISMATCH


def test_lane_requires_matching_plan_digests(monkeypatch) -> None:
    slot = slot_from_request(
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
    )
    lane = GlooControlCommunicationLane(rank=0, world_size=2, root_rank=0, process_group=None, group_ranks=(0, 1))
    local = _candidate(slot_digest=slot.semantic_digest())
    remote = LocalPublicationCandidate(
        slot=local.slot,
        planner_id=local.planner_id,
        logical_plan_digest="other",
        token=local.token,
        status=local.status,
        metadata=local.metadata,
    )
    monkeypatch.setattr(
        lane,
        "_all_gather_status",
        lambda payload: [
            dict(payload),
            {
                "slot_digest": slot.semantic_digest(),
                "group_rank": 1,
                "global_rank": 1,
                "status": "READY",
                "candidate": remote.to_dict(),
            },
        ],
    )
    result = lane.poll(slot, local)
    assert result.status is PublicationPollStatus.FAILED
    assert result.details["reason"] in {"plan_digest_mismatch", "invalid_candidate_payload"}


def test_lane_rejects_candidate_slot_identity_mismatch() -> None:
    slot = slot_from_request(
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
    )
    lane = GlooControlCommunicationLane(rank=0, world_size=1, root_rank=0, process_group=None, group_ranks=(0,))
    wrong = _candidate(slot_digest=slot.semantic_digest())
    wrong = LocalPublicationCandidate(
        slot=slot_from_request(
            run_id="run",
            forward_generation=1,
            microbatch_id="mb",
            source_layer_id="9",
            target_layer_id="10",
        ),
        planner_id=wrong.planner_id,
        logical_plan_digest=wrong.logical_plan_digest,
        token=wrong.token,
        status=wrong.status,
        metadata=wrong.metadata,
    )
    result = lane.poll(slot, wrong)
    assert result.status is PublicationPollStatus.SLOT_MISMATCH


def test_lane_uses_global_root_rank_for_broadcast(monkeypatch) -> None:
    slot = slot_from_request(
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
    )
    lane = GlooControlCommunicationLane(rank=2, world_size=2, root_rank=2, process_group=object(), group_ranks=(2, 3))
    local = _candidate(slot_digest=slot.semantic_digest())
    monkeypatch.setattr(lane_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(lane_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        lane,
        "_all_gather_status",
        lambda payload: [
            {"slot_digest": slot.semantic_digest(), "group_rank": 0, "global_rank": 2, "status": "READY", "candidate": local.to_dict()},
            {"slot_digest": slot.semantic_digest(), "group_rank": 1, "global_rank": 3, "status": "READY", "candidate": local.to_dict()},
        ],
    )
    seen = {}

    def _broadcast(object_list, *, src, group):
        seen["src"] = src
        seen["group"] = group

    monkeypatch.setattr(lane_mod.dist, "broadcast_object_list", _broadcast)
    result = lane.poll(slot, local)
    assert result.status is PublicationPollStatus.READY
    assert seen["src"] == 2


def test_lane_generation_floor_marks_old_slot_expired() -> None:
    slot = slot_from_request(
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
    )
    lane = GlooControlCommunicationLane(rank=0, world_size=1, root_rank=0, process_group=None, group_ranks=(0,))
    lane.cancel_before_generation(run_id="run", microbatch_id="mb", current_generation=3)
    result = lane.poll(slot, None)
    assert result.status is PublicationPollStatus.EXPIRED
