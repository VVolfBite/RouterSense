from __future__ import annotations

from rs.runtime.offline.replay_fixture import (
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


def test_replay_fixture_builder_zeros_diagonal_but_keeps_self_metadata() -> None:
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
            "per_rank_peer_bytes": [1000, 64],
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
            "per_rank_peer_bytes": [32, 2000],
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
            "per_rank_peer_bytes": [3000, 16],
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
            "per_rank_peer_bytes": [8, 4000],
        },
    ]
    bundle = build_replay_fixture_bundle(rows, policy_name="disabled")
    fixture = bundle["fixtures"][0]
    assert fixture["p0_dispatch_matrix"] == [[0, 64], [32, 0]]
    assert fixture["p1_return_matrix"] == [[0, 16], [8, 0]]
    audit = build_replay_fixture_audit_summary(bundle, source_kind="control_replay_trace", trace_file_count=2)
    layer = audit["layers"][0]
    assert layer["p0_self_bytes"] == 3000
    assert layer["p1_self_bytes"] == 7000
    assert layer["p0_total_bytes"] == 96
    assert layer["p1_total_bytes"] == 24


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
