from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
