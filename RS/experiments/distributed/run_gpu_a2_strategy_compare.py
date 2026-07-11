#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distributed._gpu_runner_common import (
    available_cuda_count,
    build_strategy_comparison_config,
    copy_config,
    dump_yaml,
    load_yaml,
    read_json,
    run_subprocess,
    write_json,
)


DEFAULT_STRATEGIES = (
    "native",
    "fifo_async_p2p",
    "greedy_async_p2p",
    "birkhoff_phase_local_sync",
    "birkhoff_phase_local_async_p2p",
    "routersense_joint_phase_sync",
    "routersense_joint_zero_hint_async_p2p",
    "routersense_joint_predicted_async_p2p",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4GPU A2 comparison body or an explicit no-4GPU fallback.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategies", nargs="*", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=7)
    parser.add_argument("--selected-layers", default="all")
    parser.add_argument("--profile", default="perf", choices=("debug", "execution", "perf"))
    parser.add_argument("--preflight-mode", default="compact", choices=("full", "compact"))
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _fallback(output_dir: Path, *, world_size: int, config: str, strategies: list[str], warmup_iters: int, measure_iters: int, selected_layers: str, profile: str, preflight_mode: str, dry_run: bool) -> dict:
    gate_cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py",
    ]
    proc = run_subprocess(gate_cmd)
    payload = {
        "runner": "run_gpu_a2_strategy_compare",
        "config": str(config),
        "strategies": list(strategies),
        "warmup_iters": int(warmup_iters),
        "measure_iters": int(measure_iters),
        "selected_layers": str(selected_layers),
        "profile": str(profile),
        "preflight_mode": str(preflight_mode),
        "world_size": int(world_size),
        "dry_run": bool(dry_run),
        "status": "IMPLEMENTED_GPU_BLOCKED_BY_ENVIRONMENT",
        "fallback_used": True,
        "result_eligible_for_performance_comparison": False,
        "fallback_reason": "gpu_environment_insufficient_world_size",
        "fallback_command": gate_cmd,
        "fallback_returncode": int(proc.returncode),
    }
    (output_dir / "fallback_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "fallback_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return payload


def _safe_delta(report: dict, left: str, right: str) -> float | None:
    by_name = {str(item.get("name")): item for item in report.get("strategies", []) or []}
    left_row = by_name.get(left, {})
    right_row = by_name.get(right, {})
    left_us = ((left_row.get("metrics") or {}).get("total_forward_us"))
    right_us = ((right_row.get("metrics") or {}).get("total_forward_us"))
    if left_us is None or right_us in (None, 0):
        return None
    return float((float(left_us) - float(right_us)) / float(right_us))


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(Path(args.config), output_dir)
    base = load_yaml(Path(args.config))
    payload = {
        "runner": "run_gpu_a2_strategy_compare",
        "config": str(args.config),
        "strategies": list(args.strategies),
        "warmup_iters": int(args.warmup_iters),
        "measure_iters": int(args.measure_iters),
        "selected_layers": str(args.selected_layers),
        "profile": str(args.profile),
        "preflight_mode": str(args.preflight_mode),
        "world_size": int(args.world_size),
        "dry_run": bool(args.dry_run),
        "derived_metrics": [
            "p2p_backend_gain",
            "joint_gain",
            "prediction_gain",
            "full_system_gain",
        ],
    }
    if args.dry_run:
        payload["status"] = "dry_run_ready"
        write_json(output_dir / "a2_runner_summary.json", payload)
        print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
        return 0
    if available_cuda_count() < int(args.world_size):
        payload = _fallback(
            output_dir,
            world_size=int(args.world_size),
            config=str(args.config),
            strategies=list(args.strategies),
            warmup_iters=int(args.warmup_iters),
            measure_iters=int(args.measure_iters),
            selected_layers=str(args.selected_layers),
            profile=str(args.profile),
            preflight_mode=str(args.preflight_mode),
            dry_run=bool(args.dry_run),
        )
        write_json(output_dir / "a2_runner_summary.json", payload)
        print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
        return 0 if int(payload["fallback_returncode"]) == 0 else int(payload["fallback_returncode"])

    repetitions = int(args.warmup_iters) + int(args.measure_iters)
    comparison = build_strategy_comparison_config(
        base_comparison=base,
        strategies=[str(item) for item in args.strategies],
        repetitions=repetitions,
        profile=str(args.profile),
        output_mode="paper" if str(args.profile) == "perf" else "debug_replay",
    )
    generated_config = output_dir / "generated_configs" / "a2_comparison.yaml"
    dump_yaml(generated_config, comparison)
    cmd = [
        "python",
        "-m",
        "experiments.online.run_strategy_comparison",
        "--config",
        str(generated_config),
        "--output-dir",
        str(output_dir / "comparison_run"),
    ]
    proc = run_subprocess(cmd)
    (output_dir / "comparison_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "comparison_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        payload.update({"status": "comparison_failed", "comparison_command": cmd, "comparison_returncode": int(proc.returncode)})
        write_json(output_dir / "a2_runner_summary.json", payload)
        print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
        return int(proc.returncode)
    report = read_json(output_dir / "comparison_run" / "comparison_report.json")
    payload.update(
        {
            "status": "executed",
            "comparison_command": cmd,
            "comparison_report_path": str(output_dir / "comparison_run" / "comparison_report.json"),
            "p2p_backend_gain": _safe_delta(report, "birkhoff_phase_local_async_p2p", "birkhoff_phase_local_sync"),
            "joint_gain": _safe_delta(report, "routersense_joint_zero_hint_async_p2p", "birkhoff_phase_local_async_p2p"),
            "prediction_gain": _safe_delta(report, "routersense_joint_predicted_async_p2p", "routersense_joint_zero_hint_async_p2p"),
            "full_system_gain": _safe_delta(report, "routersense_joint_predicted_async_p2p", "birkhoff_phase_local_sync"),
        }
    )
    write_json(output_dir / "a2_runner_summary.json", payload)
    print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
