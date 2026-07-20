from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    AsyncReleaseP2PExecutor,
    AsyncReleaseP2PExecutorConfig,
    AsyncReleasePlanBuilder,
    AsyncReleaseRankContext,
)
from rs.scheduling.online_adapters.plan_priority import PlanPriorityArtifact, PriorityEntry


def _artifact() -> PlanPriorityArtifact:
    return PlanPriorityArtifact(
        source_policy="safe_pair",
        joint_policy="future:p012:joint:global:rscf",
        local_policy="future:p012:local:global:rscf",
        selected_policy="future:p012:joint:global:rscf",
        fallback_to_local=False,
        heuristic_family="rscf",
        predictor_name="copy_current_dispatch",
        p2_source="copy_current_dispatch",
        priority_entries=(
            PriorityEntry("p0_dispatch", 0, 1, 8, 10.0, 0, 8, "none"),
            PriorityEntry("p1_return", 1, 0, 4, 9.0, 1, 4, "wait_p0_complete"),
        ),
    )


def test_async_release_p2p_executor_orders_recvs_before_sends_deterministically() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=False).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0"},
    )
    executor = AsyncReleaseP2PExecutor(config=AsyncReleaseP2PExecutorConfig(enabled=True))
    report = executor.execute(plan)
    ordered = report["ordered_ops"]
    assert ordered[0]["op_kind"] == "recv"
    assert ordered[1]["op_kind"] == "send"
    assert report["real_collectives_executed"] is False
    assert report["fallback_to_phase_sync"] is True


def test_async_release_p2p_executor_filters_ops_by_local_rank() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=False).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0"},
    )
    executor = AsyncReleaseP2PExecutor(config=AsyncReleaseP2PExecutorConfig(enabled=True))
    report_rank0 = executor.execute(plan, rank_context=AsyncReleaseRankContext(global_rank=0, local_rank=0, ep_group_ranks=(0, 1)))
    report_rank1 = executor.execute(plan, rank_context=AsyncReleaseRankContext(global_rank=1, local_rank=1, ep_group_ranks=(0, 1)))
    rank0_ops = report_rank0["ordered_ops"]
    rank1_ops = report_rank1["ordered_ops"]
    assert [item["op_kind"] for item in rank0_ops] == ["send", "recv"]
    assert [item["peer_rank"] for item in rank0_ops] == [1, 1]
    assert [item["op_kind"] for item in rank1_ops] == ["recv", "send"]
    assert [item["peer_rank"] for item in rank1_ops] == [0, 0]


def test_async_release_p2p_executor_distinguishes_global_and_local_rank_namespaces() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=False).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0"},
    )
    executor = AsyncReleaseP2PExecutor(config=AsyncReleaseP2PExecutorConfig(enabled=True))
    rank2 = executor.execute(
        plan,
        rank_context=AsyncReleaseRankContext(global_rank=2, local_rank=0, ep_group_ranks=(2, 3)),
    )
    rank3 = executor.execute(
        plan,
        rank_context=AsyncReleaseRankContext(global_rank=3, local_rank=1, ep_group_ranks=(2, 3)),
    )
    # This fixture has tasks between global ranks 0 and 1, so ranks 2/3 are non-participants.
    assert rank2["op_count"] == 0
    assert rank3["op_count"] == 0
