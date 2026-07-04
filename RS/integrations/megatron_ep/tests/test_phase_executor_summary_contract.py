from __future__ import annotations

from integrations.megatron_ep.routersense.contracts import NativeEPSummary


def test_phase_executor_summary_contract() -> None:
    payload = NativeEPSummary(
        ep_size=2,
        dispatcher="alltoall",
        backend="nccl",
        details={
            "rank_summaries": [
                {
                    "rank": 0,
                    "device": "cuda:0",
                    "legacy_scheduler_mode": "bucketed_fifo",
                    "policy_name": "bucketed_fifo",
                    "policy_version": "v1",
                    "execution_mode": "phase_sync_wave",
                    "control_mode": "sync_before_phase",
                    "transport_mutation": True,
                }
            ]
        },
    ).to_dict()
    rank_summaries = payload["details"]["rank_summaries"]
    assert isinstance(rank_summaries, list)
    assert rank_summaries[0]["legacy_scheduler_mode"] == "bucketed_fifo"
    assert rank_summaries[0]["policy_name"] == "bucketed_fifo"
    assert rank_summaries[0]["execution_mode"] == "phase_sync_wave"
    assert rank_summaries[0]["control_mode"] == "sync_before_phase"
    assert rank_summaries[0]["transport_mutation"] is True
