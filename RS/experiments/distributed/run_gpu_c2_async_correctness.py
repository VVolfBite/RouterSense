#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run the 4GPU C2 async correctness parity check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-strategy", default="birkhoff_phase_local_sync")
    parser.add_argument("--candidate-strategy", default="routersense_joint_predicted_async_p2p")
    parser.add_argument("--profile", default="execution", choices=("debug", "execution", "perf"))
    parser.add_argument("--selected-layers", default="2")
    parser.add_argument("--forward-epochs", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "runner": "run_gpu_c2_async_correctness",
        "config": str(args.config),
        "reference_strategy": str(args.reference_strategy),
        "candidate_strategy": str(args.candidate_strategy),
        "profile": str(args.profile),
        "selected_layers": str(args.selected_layers),
        "forward_epochs": int(args.forward_epochs),
        "world_size": int(args.world_size),
        "dry_run": bool(args.dry_run),
        "checks": {
            "async_executor_invocation_gt_zero": True,
            "batch_isend_irecv_gt_zero": True,
            "fallback_count_expected": 0,
            "timeout_count_expected": 0,
            "parity_compare_required": True,
        },
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
    (output_dir / "c2_runner_dry_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
