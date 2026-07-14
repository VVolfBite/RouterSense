#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distributed._gpu_runner_common import (
    available_cuda_count,
    build_policy_correctness_config,
    copy_config,
    dump_yaml,
    load_official_config,
    read_json,
    run_subprocess,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native vs RouterSense single-GPU flow correctness.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _compare_logits(native_path: Path, candidate_path: Path) -> dict[str, object]:
    native = torch.load(native_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    same_shape = list(native.shape) == list(candidate.shape)
    same_dtype = str(native.dtype) == str(candidate.dtype)
    diff = (native.float() - candidate.float()).abs()
    return {
        "shape_equal": same_shape,
        "dtype_equal": same_dtype,
        "allclose": bool(torch.allclose(native.float(), candidate.float(), rtol=1e-4, atol=1e-4)),
        "max_abs_error": float(diff.max().item()) if diff.numel() else 0.0,
        "nan_count": int(torch.isnan(candidate).sum().item()),
        "inf_count": int(torch.isinf(candidate).sum().item()),
    }


def _summary_from_run(run_dir: Path) -> dict[str, object]:
    return read_json(run_dir / "summary.json")


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path).resolve()
    copy_config(Path(args.config), output_dir)

    cuda_count = available_cuda_count()
    if cuda_count < 1:
        payload = {
            "status": "environment_not_run",
            "reason": "cuda_device_count_lt_1",
            "cuda_device_count": int(cuda_count),
            "transport_validation_scope": "single_rank_no_remote_transport",
        }
        write_json(summary_path, payload)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return 0

    base = load_official_config(REPO_ROOT / "configs" / "online_phase_sync.yaml")
    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    native_root = output_dir / "native"
    routersense_root = output_dir / "routersense"

    native_config = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="native",
        run_name="singlecard_native",
        output_root=native_root,
        profile="execution",
        selected_layers="0,1",
        save_logits=True,
        preflight_mode="compact",
    )
    native_config["topology"]["launcher"]["nproc_per_node"] = 1
    native_config["topology"]["ep"]["size"] = 1
    native_config["execution"]["bucket_rows"] = 0
    native_config_path = generated_dir / "singlecard_native.yaml"
    dump_yaml(native_config_path, native_config)

    routersense_config = build_policy_correctness_config(
        base_comparison=base,
        strategy_name="routersense_joint_phase_sync",
        run_name="singlecard_routersense",
        output_root=routersense_root,
        profile="execution",
        selected_layers="0,1",
        save_logits=True,
        preflight_mode="compact",
    )
    routersense_config["topology"]["launcher"]["nproc_per_node"] = 1
    routersense_config["topology"]["ep"]["size"] = 1
    routersense_config["runtime"]["line"] = "phase_sync"
    routersense_config["execution"]["mode"] = "phase_sync_wave"
    routersense_config["execution"]["bucket_rows"] = 0
    routersense_config_path = generated_dir / "singlecard_routersense.yaml"
    dump_yaml(routersense_config_path, routersense_config)

    native_cmd = [
        sys.executable,
        "-m",
        "experiments.online.collect_native_ep_trace",
        "--config",
        str(native_config_path),
        "--run-id",
        "singlecard_native",
        "--output-dir",
        str(native_root),
    ]
    native_proc = run_subprocess(native_cmd, extra_env={"ROUTERSENSE_WARMUP_ITERS": "0", "ROUTERSENSE_MEASURE_ITERS": "1"})
    (output_dir / "native.stdout.log").write_text(native_proc.stdout, encoding="utf-8")
    (output_dir / "native.stderr.log").write_text(native_proc.stderr, encoding="utf-8")
    if native_proc.returncode != 0:
        native_summary_path = native_root / "singlecard_native" / "summary.json"
        if native_summary_path.is_file():
            native_summary = read_json(native_summary_path)
            if str(native_summary.get("status", "")) == "blocked_environment":
                payload = {
                    "status": "environment_not_run",
                    "reason": str(native_summary.get("reason", "blocked_environment")),
                    "environment_summary_path": str(native_summary_path),
                    "environment_details": native_summary,
                    "transport_validation_scope": "single_rank_no_remote_transport",
                }
                write_json(summary_path, payload)
                print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
                return 0
        payload = {
            "status": "failure",
            "reason": "native_run_failed",
            "returncode": int(native_proc.returncode),
            "transport_validation_scope": "single_rank_no_remote_transport",
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 1,
        }
        write_json(summary_path, payload)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return int(native_proc.returncode)

    routersense_cmd = [
        sys.executable,
        "-m",
        "experiments.online.run_policy_correctness",
        "--config",
        str(routersense_config_path),
        "--run-id",
        "singlecard_routersense",
        "--output-dir",
        str(routersense_root),
    ]
    routersense_proc = run_subprocess(
        routersense_cmd,
        extra_env={"ROUTERSENSE_WARMUP_ITERS": "0", "ROUTERSENSE_MEASURE_ITERS": "1"},
    )
    (output_dir / "routersense.stdout.log").write_text(routersense_proc.stdout, encoding="utf-8")
    (output_dir / "routersense.stderr.log").write_text(routersense_proc.stderr, encoding="utf-8")
    if routersense_proc.returncode != 0:
        routersense_summary_path = routersense_root / "singlecard_routersense" / "summary.json"
        if routersense_summary_path.is_file():
            routersense_summary = read_json(routersense_summary_path)
            if str(routersense_summary.get("status", "")) == "blocked_environment":
                payload = {
                    "status": "environment_not_run",
                    "reason": str(routersense_summary.get("reason", "blocked_environment")),
                    "environment_summary_path": str(routersense_summary_path),
                    "environment_details": routersense_summary,
                    "transport_validation_scope": "single_rank_no_remote_transport",
                }
                write_json(summary_path, payload)
                print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
                return 0
        payload = {
            "status": "failure",
            "reason": "routersense_run_failed",
            "returncode": int(routersense_proc.returncode),
            "transport_validation_scope": "single_rank_no_remote_transport",
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 1,
        }
        write_json(summary_path, payload)
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return int(routersense_proc.returncode)

    native_run_dir = native_root / "singlecard_native"
    routersense_run_dir = routersense_root / "singlecard_routersense"
    native_summary = _summary_from_run(native_run_dir)
    routersense_summary = _summary_from_run(routersense_run_dir)
    native_rank = read_json(native_run_dir / "rank0_summary.json")
    routersense_rank = read_json(routersense_run_dir / "rank0_summary.json")
    native_logits = Path(str(native_rank["logits_path"]))
    routersense_logits = Path(str(routersense_rank["logits_path"]))
    logits_comparison = _compare_logits(native_logits, routersense_logits)

    selected_p0 = int(routersense_rank.get("selected_p0_hook_count", 0) or 0)
    selected_p1 = int(routersense_rank.get("selected_p1_hook_count", 0) or 0)
    prediction_source_p0 = int(routersense_rank.get("prediction_source_p0_hook_count", 0) or 0)
    output_shape_equal = list(native_summary.get("details", {}).get("output_shape") or []) == list(
        routersense_summary.get("details", {}).get("output_shape") or []
    )
    passed = (
        bool(logits_comparison["allclose"])
        and bool(output_shape_equal)
        and selected_p0 > 0
        and selected_p1 > 0
        and prediction_source_p0 > 0
        and float(routersense_summary.get("details", {}).get("output_checksum") or 0.0) == float(
            native_summary.get("details", {}).get("output_checksum") or 0.0
        )
    )

    payload = {
        "status": "passed" if passed else "failed",
        "native_status": str(native_summary.get("status", "")),
        "routersense_status": str(routersense_summary.get("status", "")),
        "native_output_checksum": native_summary.get("details", {}).get("output_checksum"),
        "routersense_output_checksum": routersense_summary.get("details", {}).get("output_checksum"),
        "output_shape_equal": bool(output_shape_equal),
        "logits_comparison": logits_comparison,
        "selected_p0_hook_count": selected_p0,
        "selected_p1_hook_count": selected_p1,
        "prediction_source_p0_hook_count": prediction_source_p0,
        "planning_count": int(routersense_summary.get("details", {}).get("prediction_audit_count", 0) or 0),
        "publication_count": int(routersense_summary.get("details", {}).get("prepared_plan_binding_count", 0) or 0),
        "measurement_event_count": 0,
        "fallback_count": 0 if passed else 1,
        "timeout_count": 0,
        "check_failure_count": 0 if passed else 1,
        "transport_validation_scope": "single_rank_no_remote_transport",
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }
    write_json(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
