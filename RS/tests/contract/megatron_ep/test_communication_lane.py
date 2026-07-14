from __future__ import annotations

from rs.runtime.online.megatron_ep.control.communication_lane import GlooControlCommunicationLane, slot_from_request
from rs.runtime.online.megatron_ep.public_types import (
    LocalPreparationToken,
    LocalPublicationCandidate,
    PublicationPollStatus,
)
from rs.runtime.online.megatron_ep.target_planning.contracts import TargetLayerPreparedJointPlan
from rs.scheduling.contracts import LogicalSchedulePlan


def _candidate(*, slot_digest: str, logical_plan_digest: str = "ld") -> LocalPublicationCandidate:
    slot = slot_from_request(
        run_id="run",
        forward_generation=1,
        microbatch_id="mb",
        source_layer_id="0",
        target_layer_id="1",
    )
    plan = TargetLayerPreparedJointPlan(
        source_layer_id="0",
        target_layer_id="1",
        run_id="run",
        forward_epoch=1,
        microbatch_id="mb",
        h1_prediction_digest="h1",
        h2_prediction_digest="h2",
        target_problem_digest="tp",
        logical_plan=LogicalSchedulePlan(policy_name="u", waves=(), diagnostics={}),
        logical_plan_digest=logical_plan_digest,
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
        logical_plan_digest=logical_plan_digest,
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
    local = _candidate(slot_digest=slot.semantic_digest(), logical_plan_digest="root")
    remote = _candidate(slot_digest=slot.semantic_digest(), logical_plan_digest="other")
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
    assert result.details["reason"] == "plan_digest_mismatch"
