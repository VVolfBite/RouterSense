#!/usr/bin/env python3
"""CPU-only trace/fixture simulator for phase barrier vs streaming release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import write_json
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from rs.scheduling.multiphase.streaming_simulator import compare_barrier_and_streaming


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="streaming_release_sim")
    parser.add_argument("--mode", choices=(EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE), default=RUNTIME_LOOKAHEAD_MODE)
    parser.add_argument("--service-granularity", choices=("wave", "chunk"), default="wave")
    parser.add_argument("--chunk-size", type=float, default=None)
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    return parser.parse_args(argv)


def _matrix(payload: dict[str, Any], key: str) -> list[list[int]]:
    if key in payload:
        return [[int(value) for value in row] for row in payload[key]]
    if key == "p2_next_dispatch_matrix" and "p2_next_dispatch_forecast_matrix" in payload:
        return [[int(value) for value in row] for row in payload["p2_next_dispatch_forecast_matrix"]]
    raise KeyError(key)


def _write_report_md(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Streaming Release Simulation",
                "",
                f"- Mode: `{report['scheduling_mode']}`",
                f"- Granularity: `{report['service_granularity']}`",
                f"- Barrier makespan: `{report['barrier']['makespan']:.6f}`",
                f"- Streaming makespan: `{report['streaming']['makespan']:.6f}`",
                f"- Makespan reduction: `{report['makespan_reduction_pct']:.2f}%`",
                f"- Speedup: `{report['speedup']:.4f}x`",
                "",
                "## P1 Release Savings By Rank",
                "",
                "| Rank | Savings |",
                "| ---: | ------: |",
                *[
                    f"| {rank} | {float(value):.6f} |"
                    for rank, value in enumerate(report["p1_release_savings_by_rank"])
                ],
                "",
                "This is a CPU logical simulation. It is not GPU runtime latency.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture_path = Path(args.fixture)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    p0 = _matrix(payload, "p0_dispatch_matrix")
    p1 = _matrix(payload, "p1_return_matrix")
    p2 = _matrix(payload, "p2_next_dispatch_matrix")
    num_gpus = int(payload.get("num_gpus", len(p0)))
    report = compare_barrier_and_streaming(
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_matrix=p2,
        num_gpus=num_gpus,
        scheduling_mode=args.mode,
        expert_compute_delay=float(args.expert_compute_delay),
        service_granularity=args.service_granularity,
        chunk_size=args.chunk_size,
    )
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": args.run_id,
            "run_kind": "streaming_release_simulator",
            "fixture": str(fixture_path),
            "mode": args.mode,
            "service_granularity": args.service_granularity,
            "chunk_size": args.chunk_size,
            "expert_compute_delay": args.expert_compute_delay,
            "gpu_required": False,
            "megatron_required": False,
        },
    )
    write_json(run_dir / "streaming_release_report.json", report)
    write_json(run_dir / "barrier_schedule.json", report["barrier_schedule"])
    write_json(run_dir / "streaming_schedule.json", report["streaming_schedule"])
    _write_report_md(run_dir / "streaming_release_report.md", report)
    return 0 if bool(report["barrier"]["audit"]["valid"]) and bool(report["streaming"]["audit"]["valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
