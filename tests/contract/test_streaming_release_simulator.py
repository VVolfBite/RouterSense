from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rs.scheduling.multiphase.streaming_simulator import compare_barrier_and_streaming

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> dict:
    fixture = REPO_ROOT / "tests/fixtures/scheduling/p1_release_sensitive_4rank.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def _p2(payload: dict) -> list[list[int]]:
    return payload.get("p2_next_dispatch_matrix") or payload["p2_next_dispatch_forecast_matrix"]


def test_release_aware_streaming_releases_p1_earlier_than_barrier() -> None:
    payload = _fixture()
    report = compare_barrier_and_streaming(
        p0_dispatch_matrix=payload["p0_dispatch_matrix"],
        p1_return_matrix=payload["p1_return_matrix"],
        p2_next_dispatch_matrix=_p2(payload),
        num_gpus=payload["num_gpus"],
        scheduling_mode="runtime_lookahead",
        expert_compute_delay=2.0,
        service_granularity="chunk",
        chunk_size=8.0,
    )
    assert report["barrier"]["audit"]["valid"] is True
    assert report["streaming"]["audit"]["valid"] is True
    assert max(report["p1_release_savings_by_rank"]) > 0.0
    assert report["streaming"]["makespan"] <= report["barrier"]["makespan"]


def test_streaming_release_cli_writes_report(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(_fixture()), encoding="utf-8")
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.run_streaming_release_simulator",
            "--fixture",
            str(fixture_path),
            "--output-dir",
            str(out),
            "--run-id",
            "case",
            "--mode",
            "runtime_lookahead",
            "--service-granularity",
            "chunk",
            "--chunk-size",
            "8",
            "--expert-compute-delay",
            "2",
        ],
        check=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    report = json.loads((out / "case" / "streaming_release_report.json").read_text(encoding="utf-8"))
    assert report["service_granularity"] == "chunk"
    assert "barrier_schedule" in report
    assert "streaming_schedule" in report
    assert (out / "case" / "streaming_release_report.md").exists()
