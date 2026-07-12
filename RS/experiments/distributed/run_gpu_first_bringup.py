#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distributed._gpu_runner_common import (
    available_cuda_count,
    copy_config,
    load_official_config,
    run_subprocess,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first 4GPU bring-up workflow in strict staged order.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(config_path, output_dir)
    config = load_official_config(config_path)
    stages = [
        {"stage": "environment_preflight", "runner": "", "config": str(config_path)},
        {"stage": "native_smoke", "runner": "experiments/distributed/run_gpu_b2_lifecycle.py", "config": str(config_path)},
        {"stage": "phase_sync_smoke", "runner": "experiments/distributed/run_gpu_b2_lifecycle.py", "config": str(config_path)},
        {"stage": "async_b_core_smoke", "runner": "experiments/distributed/run_gpu_b2_lifecycle.py", "config": str(config_path)},
        {"stage": "u_zero_smoke", "runner": "experiments/distributed/run_gpu_b2_lifecycle.py", "config": str(config_path)},
        {"stage": "u_predicted_ready_smoke", "runner": "experiments/distributed/run_gpu_b2_lifecycle.py", "config": str(config_path)},
        {"stage": "late_plan_smoke", "runner": "experiments/distributed/run_target_plan_hotpath_gloo_gate.py", "config": str(config_path)},
        {"stage": "safe_u_smoke", "runner": "experiments/distributed/run_gpu_b2_lifecycle.py", "config": str(config_path)},
        {"stage": "c2", "runner": "experiments/distributed/run_gpu_c2_async_correctness.py", "config": str(REPO_ROOT / 'configs/official/gpu_c2_correctness.yaml')},
        {"stage": "short_a2", "runner": "experiments/distributed/run_gpu_a2_strategy_compare.py", "config": str(REPO_ROOT / 'configs/official/gpu_first_bringup.yaml')},
    ]
    payload = {
        "runner": "run_gpu_first_bringup",
        "config": str(config_path),
        "world_size": int(config.get("world_size", (config.get("topology", {}) or {}).get("world_size", 4))),
        "selected_layers": str(config.get("selected_layers", "0,1")),
        "stages": stages,
        "cuda_device_count": int(available_cuda_count()),
    }
    if int(payload["cuda_device_count"]) < 4:
        payload.update(
            {
                "status": "gpu_environment_insufficient",
                "earliest_failure_stage": "environment_preflight",
                "concrete_failure": f"cuda_device_count={payload['cuda_device_count']} < 4",
                "last_valid_artifact": "",
            }
        )
        write_json(output_dir / "gpu_first_bringup_plan.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    stage_results: list[dict[str, object]] = []
    for stage in stages[1:]:
        runner = str(stage["runner"])
        cmd = [sys.executable, runner, "--config", str(stage["config"]), "--output-dir", str(output_dir / stage["stage"])]
        proc = run_subprocess(cmd)
        stage_result = {
            "stage": str(stage["stage"]),
            "returncode": int(proc.returncode),
            "stdout_path": str(output_dir / f"{stage['stage']}.stdout.log"),
            "stderr_path": str(output_dir / f"{stage['stage']}.stderr.log"),
        }
        Path(stage_result["stdout_path"]).write_text(proc.stdout, encoding="utf-8")
        Path(stage_result["stderr_path"]).write_text(proc.stderr, encoding="utf-8")
        stage_results.append(stage_result)
        if int(proc.returncode) != 0:
            payload.update(
                {
                    "status": "failed",
                    "earliest_failure_stage": str(stage["stage"]),
                    "concrete_failure": f"{runner} exited with {proc.returncode}",
                    "last_valid_artifact": str(output_dir / stage["stage"]),
                    "stage_results": stage_results,
                }
            )
            write_json(output_dir / "gpu_first_bringup_plan.json", payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return int(proc.returncode)
    payload.update({"status": "passed", "stage_results": stage_results})
    write_json(output_dir / "gpu_first_bringup_plan.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
