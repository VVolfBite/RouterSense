from __future__ import annotations

from experiments.offline.build_replay_fixture_from_control_trace import (
    build_replay_fixture_audit_summary,
    build_replay_fixture_bundle,
)


def test_replay_fixture_audit_detects_complete_ranks() -> None:
    rows = [
        {
            "run_id_digest": "run1",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P0",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 64],
        },
        {
            "run_id_digest": "run1",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P0",
            "global_rank": 1,
            "local_rank": 1,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [32, 0],
        },
        {
            "run_id_digest": "run1",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P1",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 16],
        },
        {
            "run_id_digest": "run1",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P1",
            "global_rank": 1,
            "local_rank": 1,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [8, 0],
        },
    ]
    bundle = build_replay_fixture_bundle(rows, policy_name="disabled")
    audit = build_replay_fixture_audit_summary(bundle, source_kind="control_replay_trace", trace_file_count=2)
    assert audit["layer_count_with_complete_p0p1"] == 1
    assert audit["layer_count_with_missing_rank"] == 0
    assert audit["layers"][0]["p0_missing_ranks"] == []
    assert audit["layers"][0]["p1_missing_ranks"] == []


def test_replay_fixture_audit_detects_missing_rank_and_last_layer_zero_p2() -> None:
    rows = [
        {
            "run_id_digest": "run1",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P0",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 64],
        },
        {
            "run_id_digest": "run1",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P1",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 16],
        },
    ]
    bundle = build_replay_fixture_bundle(rows, policy_name="disabled")
    audit = build_replay_fixture_audit_summary(bundle, source_kind="phase_context_fallback", trace_file_count=0)
    assert audit["source_kind"] == "phase_context_fallback"
    assert audit["layer_count_with_missing_rank"] == 1
    assert audit["layers"][0]["p2_source"] == "zero_for_last_layer"
    assert audit["layers"][0]["p0_missing_ranks"] == [1]
