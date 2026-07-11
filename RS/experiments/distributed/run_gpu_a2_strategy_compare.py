#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch


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
    parser = argparse.ArgumentParser(description="Prepare or run the 4GPU A2 strategy comparison.")
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


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        "status": "dry_run_ready" if args.dry_run else "ready_for_gpu_execution",
    }
    if not args.dry_run and int(torch.cuda.device_count()) < int(args.world_size):
        gate_cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            "experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py",
        ]
        proc = subprocess.run(gate_cmd, cwd=str(Path(__file__).resolve().parents[2]), capture_output=True, text=True, check=False)
        payload.update(
            {
                "status": "implemented_with_tested_fallback",
                "result_eligible_for_performance_comparison": False,
                "fallback_used": True,
                "fallback_reason": "gpu_environment_insufficient_world_size",
                "fallback_command": gate_cmd,
                "fallback_returncode": int(proc.returncode),
            }
        )
    (output_dir / "a2_runner_dry_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
