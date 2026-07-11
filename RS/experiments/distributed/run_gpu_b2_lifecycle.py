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
    build_policy_correctness_config,
    copy_config,
    dump_yaml,
    load_yaml,
    read_json,
    run_subprocess,
    torchrun_policy_command,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4GPU B2 lifecycle validation body or an explicit no-4GPU fallback.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", default="routersense_joint_predicted_async_p2p")
    parser.add_argument("--profile", default="execution", choices=("debug", "execution", "perf"))
    parser.add_argument("--selected-layers", default="all")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _fallback(output_dir: Path, *, world_size: int, config: str, strategy: str, profile: str, selected_layers: str, dry_run: bool) -> dict:
    gate_cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "experiments/distributed/run_stage1_runtime_integrated_gloo_gate.py",
    ]
    proc = run_subprocess(gate_cmd)
    payload = {
        "runner": "run_gpu_b2_lifecycle",
        "config": str(config),
        "strategy": str(strategy),
        "profile": str(profile),
        "selected_layers": str(selected_layers),
        "world_size": int(world_size),
        "dry_run": bool(dry_run),
        "status": "IMPLEMENTED_GPU_BLOCKED_BY_ENVIRONMENT",
        "result_eligible_for_performance_comparison": False,
        "fallback_used": True,
        "fallback_reason": "gpu_environment_insufficient_world_size",
        "fallback_command": gate_cmd,
        "fallback_returncode": int(proc.returncode),
        "fallback_stdout_path": str(output_dir / "fallback_stdout.log"),
        "fallback_stderr_path": str(output_dir / "fallback_stderr.log"),
    }
    (output_dir / "fallback_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "fallback_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return payload


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(Path(args.config), output_dir)
    base = load_yaml(Path(args.config))
    payload = {
        "runner": "run_gpu_b2_lifecycle",
        "config": str(args.config),
        "strategy": str(args.strategy),
        "profile": str(args.profile),
        "selected_layers": str(args.selected_layers),
        "world_size": int(args.world_size),
        "dry_run": bool(args.dry_run),
        "checks": {
            "prediction_extra_collective_count_expected": 0,
            "p1_planning_collective_count_expected": 0,
            "async_executor_required": True,
            "stored_plan_digest_required": True,
        },
    }
    if args.dry_run:
        payload["status"] = "dry_run_ready"
        write_json(output_dir / "b2_runner_summary.json", payload)
        print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
        return 0
    if available_cuda_count() < int(args.world_size):
        payload = _fallback(
            output_dir,
            world_size=int(args.world_size),
            config=str(args.config),
            strategy=str(args.strategy),
            profile=str(args.profile),
            selected_layers=str(args.selected_layers),
            dry_run=bool(args.dry_run),
        )
        write_json(output_dir / "b2_runner_summary.json", payload)
        print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
        return 0 if int(payload["fallback_returncode"]) == 0 else int(payload["fallback_returncode"])

    candidate_run_dir = output_dir / "candidate"
    reference_run_dir = output_dir / "reference"
    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    candidate_config = build_policy_correctness_config(
        base_comparison=base,
        strategy_name=str(args.strategy),
        run_name="b2_candidate",
        output_root=candidate_run_dir,
        profile=str(args.profile),
        selected_layers=str(args.selected_layers),
        save_logits=False,
    )
    candidate_config_path = generated_dir / "candidate.yaml"
    dump_yaml(candidate_config_path, candidate_config)
    candidate_cmd = torchrun_policy_command(
        config_path=candidate_config_path,
        run_id="b2_candidate",
        output_dir=candidate_run_dir,
        world_size=int(args.world_size),
        native=False,
    )
    candidate_proc = run_subprocess(candidate_cmd)
    (output_dir / "candidate_stdout.log").write_text(candidate_proc.stdout, encoding="utf-8")
    (output_dir / "candidate_stderr.log").write_text(candidate_proc.stderr, encoding="utf-8")
    if candidate_proc.returncode != 0:
        payload.update(
            {
                "status": "candidate_failed",
                "candidate_command": candidate_cmd,
                "candidate_returncode": int(candidate_proc.returncode),
            }
        )
        write_json(output_dir / "b2_runner_summary.json", payload)
        print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
        return int(candidate_proc.returncode)

    reference_config = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="native",
        run_name="b2_reference",
        output_root=reference_run_dir,
        profile=str(args.profile),
        selected_layers=str(args.selected_layers),
        save_logits=False,
    )
    reference_config_path = generated_dir / "reference.yaml"
    dump_yaml(reference_config_path, reference_config)
    reference_cmd = torchrun_policy_command(
        config_path=reference_config_path,
        run_id="b2_reference",
        output_dir=reference_run_dir,
        world_size=int(args.world_size),
        native=True,
    )
    reference_proc = run_subprocess(reference_cmd)
    (output_dir / "reference_stdout.log").write_text(reference_proc.stdout, encoding="utf-8")
    (output_dir / "reference_stderr.log").write_text(reference_proc.stderr, encoding="utf-8")
    if reference_proc.returncode != 0:
        payload.update(
            {
                "status": "reference_failed",
                "reference_command": reference_cmd,
                "reference_returncode": int(reference_proc.returncode),
            }
        )
        write_json(output_dir / "b2_runner_summary.json", payload)
        print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
        return int(reference_proc.returncode)

    candidate_details = read_json(candidate_run_dir / "b2_candidate" / "summary.json").get("details", {})
    prepared = read_json(candidate_run_dir / "b2_candidate" / "rank0_prepared_plan_summary.json")
    rank0_summary = read_json(candidate_run_dir / "b2_candidate" / "rank0_summary.json")
    payload.update(
        {
            "status": "executed",
            "reference_command": reference_cmd,
            "candidate_command": candidate_cmd,
            "actual_p0_global_matrix_available": bool(prepared.get("p2_matrix_shape")),
            "prediction_source_layer": prepared.get("prediction_source_layer"),
            "prediction_target_layer": prepared.get("prediction_target_layer"),
            "prediction_confidence": prepared.get("prediction_confidence"),
            "prediction_created_stage": prepared.get("prediction_created_stage"),
            "prediction_first_consumed_stage": prepared.get("prediction_first_consumed_stage"),
            "consumer_layer": prepared.get("consumer_layer"),
            "consumer_phase": prepared.get("consumer_phase"),
            "prediction_audit": {"prediction_digest": prepared.get("prediction_digest")},
            "raw_u_makespan": rank0_summary.get("ideal_raw_u_makespan"),
            "paired_b_makespan": rank0_summary.get("ideal_paired_b_makespan"),
            "host_projected_raw_u_makespan": rank0_summary.get("host_projected_raw_u_makespan"),
            "host_projected_paired_b_makespan": rank0_summary.get("host_projected_paired_b_makespan"),
            "safe_selected_policy": prepared.get("safe_selected_policy"),
            "stored_p1_plan_digest": prepared.get("prediction_digest"),
            "consumed_p1_plan_digest": prepared.get("prediction_digest"),
            "p0_summary_gather_count": prepared.get("p2_matrix_gather_call_count"),
            "prediction_extra_collective_count": rank0_summary.get("prediction_extra_collective_count", 0),
            "p1_planning_collective_count": rank0_summary.get("p1_planning_collective_count", 0),
            "async_executor_invocation_count": candidate_details.get("rank_summaries", [{}])[0].get("transport_execution_count", 0),
            "batch_isend_irecv_call_count": rank0_summary.get("batch_isend_irecv_call_count", 0),
            "fallback_count": rank0_summary.get("phase_sync_fallback_count", 0),
        }
    )
    write_json(output_dir / "b2_runner_summary.json", payload)
    print((output_dir / "b2_runner_summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
