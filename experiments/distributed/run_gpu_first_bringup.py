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
    python_module_command,
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
    bringup_config = REPO_ROOT / "configs/official/gpu_first_bringup.yaml"
    c2_config = REPO_ROOT / "configs/official/gpu_c2_correctness.yaml"
    stages = [
        {"stage": "environment_preflight", "runner": "", "config": str(config_path)},
        {
            "stage": "native_smoke",
            "cmd": python_module_command(
                module="experiments.distributed.run_gpu_a2_strategy_compare",
                args=[
                    "--config", str(bringup_config),
                    "--strategies", "native",
                    "--warmup-iters", "1",
                    "--measure-iters", "1",
                    "--selected-layers", "0,1",
                    "--world-size", "4",
                    "--output-dir",
                ],
            ),
        },
        {
            "stage": "phase_sync_smoke",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_b2_lifecycle.py", "--config", str(bringup_config), "--strategy", "birkhoff_phase_local_sync", "--selected-layers", "0,1", "--world-size", "4", "--output-dir"],
        },
        {
            "stage": "async_birkhoff_smoke",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_b2_lifecycle.py", "--config", str(bringup_config), "--strategy", "birkhoff_phase_local_async_p2p", "--selected-layers", "0,1", "--world-size", "4", "--output-dir"],
        },
        {
            "stage": "async_b_core_smoke",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_b2_lifecycle.py", "--config", str(bringup_config), "--strategy", "routersense_current_p012_local_event_rscf_async", "--selected-layers", "0,1", "--world-size", "4", "--output-dir"],
        },
        {
            "stage": "u_zero_smoke",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_b2_lifecycle.py", "--config", str(bringup_config), "--strategy", "routersense_current_p012_joint_event_rscf_async", "--selected-layers", "0,1", "--world-size", "4", "--output-dir"],
        },
        {
            "stage": "u_predicted_raw_smoke",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_b2_lifecycle.py", "--config", str(bringup_config), "--strategy", "routersense_future_p012_joint_event_rscf_async", "--selected-layers", "0,1", "--world-size", "4", "--output-dir"],
        },
        {
            "stage": "u_predicted_safe_smoke",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_b2_lifecycle.py", "--config", str(bringup_config), "--strategy", "routersense_future_p012_joint_global_rscf_async", "--selected-layers", "0,1", "--world-size", "4", "--output-dir"],
        },
        {
            "stage": "gpu_target_lifecycle",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_target_plan_lifecycle_smoke.py", "--config", str(bringup_config), "--selected-layers", "0,1", "--world-size", "4", "--output-dir"],
        },
        {
            "stage": "c2",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_c2_async_correctness.py", "--config", str(c2_config), "--selected-layers", "0,1", "--world-size", "4", "--output-dir"],
        },
        {
            "stage": "short_a2",
            "cmd": [sys.executable, "experiments/distributed/run_gpu_a2_strategy_compare.py", "--config", str(bringup_config), "--warmup-iters", "1", "--measure-iters", "3", "--selected-layers", "0,1", "--world-size", "4", "--c2-summary-path", "", "--output-dir"],
        },
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
    c2_summary_path = output_dir / "c2" / "c2_runner_summary.json"
    for stage in stages[1:]:
        cmd = list(stage["cmd"])
        if "--output-dir" in cmd:
            cmd[cmd.index("--output-dir") + 1:cmd.index("--output-dir") + 1] = [str(output_dir / stage["stage"])]
        else:
            cmd.extend(["--output-dir", str(output_dir / stage["stage"])])
        if str(stage["stage"]) == "short_a2":
            empty_index = cmd.index("") if "" in cmd else -1
            if empty_index >= 0:
                cmd[empty_index] = str(c2_summary_path)
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
                    "concrete_failure": f"{' '.join(cmd[:6])} exited with {proc.returncode}",
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
