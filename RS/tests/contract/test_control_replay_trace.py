from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments.offline.build_replay_fixture_from_control_trace import build_replay_fixture_bundle
from experiments.offline.replay_online_control_trace import read_jsonl, summarize_control_replay_trace


def test_summarize_control_replay_trace() -> None:
    rows = [
        {
            "policy_name": "routersense_p0p1p2_hint",
            "phase": "P0",
            "nonzero_edge_count": 3,
            "abstract_plan_summary": {"wave_count": 3, "task_ref_count": 6},
            "timing_summary": {
                "all_gather_time_us": 10.0,
                "build_plan_time_us": 20.0,
                "broadcast_time_us": 5.0,
            },
            "transport_summary": {
                "planning_summary_tensor_len": 32,
                "abstract_plan_tensor_len": 64,
                "bucket_count": 6,
                "total_byte_count": 1024,
            },
        },
        {
            "policy_name": "routersense_p0p1p2_hint",
            "phase": "P1",
            "nonzero_edge_count": 2,
            "abstract_plan_summary": {"wave_count": 2, "task_ref_count": 4},
            "timing_summary": {
                "all_gather_time_us": 11.0,
                "build_plan_time_us": 21.0,
                "broadcast_time_us": 6.0,
            },
            "transport_summary": {
                "planning_summary_tensor_len": 33,
                "abstract_plan_tensor_len": 65,
                "bucket_count": 4,
                "total_byte_count": 2048,
            },
        },
    ]
    summary = summarize_control_replay_trace(rows)
    assert summary["total_phase_count"] == 2
    assert summary["total_all_gather_calls"] == 2
    assert summary["total_broadcast_calls"] == 2
    assert summary["total_summary_elements"] == 65
    assert summary["total_plan_elements"] == 129
    assert summary["total_task_refs"] == 10
    assert summary["avg_summary_elements_per_phase"] == 32.5
    assert summary["avg_plan_elements_per_phase"] == 64.5
    assert summary["avg_task_refs_per_phase"] == 5.0
    assert summary["avg_wave_count"] == 2.5
    assert summary["max_wave_count"] == 3
    assert summary["avg_bucket_count"] == 5.0
    assert summary["avg_nonzero_edge_count"] == 2.5
    assert summary["max_nonzero_edge_count"] == 3
    assert summary["avg_total_byte_count"] == 1536.0
    assert summary["max_total_byte_count"] == 2048
    assert summary["per_policy"]["routersense_p0p1p2_hint"]["phase_count"] == 2
    assert summary["per_phase"]["P0"]["wave_count"] == 3


def test_replay_online_control_trace_cli(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    rows = [
        {
            "policy_name": "birkhoff_phase_local",
            "phase": "P0",
            "nonzero_edge_count": 1,
            "abstract_plan_summary": {"wave_count": 1, "task_ref_count": 2},
            "timing_summary": {"all_gather_time_us": 1.0, "build_plan_time_us": 2.0, "broadcast_time_us": 3.0},
            "transport_summary": {"planning_summary_tensor_len": 4, "abstract_plan_tensor_len": 5, "bucket_count": 2, "total_byte_count": 128},
        }
    ]
    trace_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert read_jsonl(trace_path) == rows
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.replay_online_control_trace",
            "--trace",
            str(trace_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd="/root/autodl-tmp/RouterSense/RS",
        env={"PYTHONPATH": "src"},
    )
    payload = json.loads(completed.stdout)
    assert payload["total_phase_count"] == 1
    assert payload["per_policy"]["birkhoff_phase_local"]["summary_elements"] == 4


def test_build_replay_fixture_bundle_from_rank_rows() -> None:
    rows = [
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P0",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 128],
            "nonzero_edges": [{"src_rank": 0, "dst_rank": 1, "row_count": 8, "byte_count": 128}],
        },
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P0",
            "global_rank": 1,
            "local_rank": 1,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [64, 0],
            "nonzero_edges": [{"src_rank": 1, "dst_rank": 0, "row_count": 4, "byte_count": 64}],
        },
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P1",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 32],
            "nonzero_edges": [{"src_rank": 0, "dst_rank": 1, "row_count": 2, "byte_count": 32}],
        },
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P1",
            "global_rank": 1,
            "local_rank": 1,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [96, 0],
            "nonzero_edges": [{"src_rank": 1, "dst_rank": 0, "row_count": 6, "byte_count": 96}],
        },
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "1",
            "layer_name": "layer_1",
            "phase": "P0",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 256],
            "nonzero_edges": [{"src_rank": 0, "dst_rank": 1, "row_count": 16, "byte_count": 256}],
        },
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "1",
            "layer_name": "layer_1",
            "phase": "P0",
            "global_rank": 1,
            "local_rank": 1,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 0],
            "nonzero_edges": [],
        },
    ]
    bundle = build_replay_fixture_bundle(rows, policy_name="disabled")
    assert bundle["fixture_count"] == 1
    fixture = bundle["fixtures"][0]
    assert fixture["fixture_name"] == "replay_layer_0"
    assert fixture["p0_dispatch_matrix"] == [[0, 128], [64, 0]]
    assert fixture["p1_return_matrix"] == [[0, 32], [96, 0]]
    assert fixture["p2_next_dispatch_forecast_matrix"] == [[0, 256], [0, 0]]
    assert fixture["metadata"]["p0_missing_ranks"] == []
    assert fixture["metadata"]["p1_missing_ranks"] == []


def test_build_replay_fixture_cli(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    rank0 = trace_dir / "rank0_control_replay_trace.jsonl"
    rank1 = trace_dir / "rank1_control_replay_trace.jsonl"
    rank0_rows = [
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P0",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 128],
            "nonzero_edges": [{"src_rank": 0, "dst_rank": 1, "row_count": 8, "byte_count": 128}],
        },
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P1",
            "global_rank": 0,
            "local_rank": 0,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [0, 32],
            "nonzero_edges": [{"src_rank": 0, "dst_rank": 1, "row_count": 2, "byte_count": 32}],
        },
    ]
    rank1_rows = [
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P0",
            "global_rank": 1,
            "local_rank": 1,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [64, 0],
            "nonzero_edges": [{"src_rank": 1, "dst_rank": 0, "row_count": 4, "byte_count": 64}],
        },
        {
            "run_id_digest": "abc123",
            "policy_name": "disabled",
            "layer_id": "0",
            "layer_name": "layer_0",
            "phase": "P1",
            "global_rank": 1,
            "local_rank": 1,
            "ep_group_size": 2,
            "per_rank_peer_bytes": [96, 0],
            "nonzero_edges": [{"src_rank": 1, "dst_rank": 0, "row_count": 6, "byte_count": 96}],
        },
    ]
    rank0.write_text("\n".join(json.dumps(row) for row in rank0_rows), encoding="utf-8")
    rank1.write_text("\n".join(json.dumps(row) for row in rank1_rows), encoding="utf-8")
    output_dir = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.build_replay_fixture_from_control_trace",
            "--trace-dir",
            str(trace_dir),
            "--policy",
            "disabled",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd="/root/autodl-tmp/RouterSense/RS",
        env={"PYTHONPATH": "src"},
    )
    payload = json.loads(completed.stdout)
    assert payload["fixture_count"] == 1
    bundle_summary = json.loads((output_dir / "replay_fixture_bundle_summary.json").read_text(encoding="utf-8"))
    assert bundle_summary["fixture_names"] == ["replay_layer_0"]
    fixture = json.loads((output_dir / "fixtures" / "replay_layer_0.json").read_text(encoding="utf-8"))
    assert fixture["p0_dispatch_matrix"] == [[0, 128], [64, 0]]
    assert fixture["p1_return_matrix"] == [[0, 32], [96, 0]]
