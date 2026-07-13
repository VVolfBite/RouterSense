#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), text=True).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _perf_profile_audit() -> dict[str, Any]:
    rows = [
        ("timeline_jsonl", False, False, "file_io", False, "keep disabled in perf"),
        ("planning_timing_jsonl", False, True, "memory_plus_optional_file_io", False, "audit after GPU smoke"),
        ("runtime_observation_artifact", False, True, "memory", True, "retain until first GPU smoke"),
        ("phase_context_artifact", False, True, "memory", True, "retain until first GPU smoke"),
        ("transport_bundle_artifact", False, True, "memory", True, "retain until first GPU smoke"),
        ("schedule_shadow", False, False, "planning_and_file_io", False, "keep disabled in perf"),
        ("prepared_plan_shadow", False, False, "planning_and_file_io", False, "keep disabled in perf"),
        ("per_wave_timing", False, False, "per_wave_accounting", False, "keep disabled in perf"),
        ("full_preflight", False, False, "collective", False, "use compact for hot-path smoke"),
        ("debug_correctness_replay", False, False, "file_io_and_replay", False, "keep disabled in perf"),
        ("per_event_jsonl", False, False, "file_io", False, "keep disabled in perf"),
    ]
    return {
        "features": [
            {
                "feature": feature,
                "enabled_in_perf": enabled_in_perf,
                "enabled_in_execution": enabled_in_execution,
                "hot_path_cost_expected": hot_path_cost_expected,
                "required_for_correctness": required_for_correctness,
                "recommended_action": recommended_action,
            }
            for (
                feature,
                enabled_in_perf,
                enabled_in_execution,
                hot_path_cost_expected,
                required_for_correctness,
                recommended_action,
            ) in rows
        ]
    }


def _gpu_validation_manifest(*, commit: str) -> dict[str, Any]:
    return {
        "status": "HOTPATH_CPU_CONTRACT_CLOSED",
        "starting_commit": "c5212ebd08927bb2597d7141ba775ea5cbc9064c",
        "source_commit": commit,
        "config_path": "configs/official/gpu_hotpath_iteration.yaml",
        "model": "allenai/OLMoE-1B-7B-0924-Instruct",
        "world_size": 4,
        "selected_layers": [0, 1],
        "workload": "8x16",
        "expected_batch_rows": 8,
        "expected_seq_len": 16,
        "warmup": 1,
        "measure": 1,
        "profile": "perf",
        "requested_preflight_mode": "compact",
        "strategies": [
            "native",
            "routersense_b_core_independent_async",
            "routersense_u_core_zero_raw_async",
        ],
        "expected_hook_counts": {
            "selected_p0_per_rank": 4,
            "selected_p1_per_rank": 4,
            "selected_p0_all_rank": 16,
            "selected_p1_all_rank": 16,
            "prediction_source_p0_all_rank": 0,
            "none_heavy_all_rank": 0,
            "u_zero_raw_u_all_rank_upper_bound": 16,
        },
        "expected_plan_build_upper_bound": {
            "routersense_u_core_zero_raw_async": 16,
        },
        "forbidden_tests": [
            "full first 4GPU bring-up",
            "GPU C2",
            "seven-strategy A2",
            "target lifecycle matrix",
            "large workload sweep",
            "model download",
            "CUDA/NCCL environment installation",
        ],
        "gpu_count_smoke": {
            "world_size": 4,
            "selected_layers": [0, 1],
            "workload": "8x16",
            "warmup": 1,
            "measure": 1,
            "profile": "perf",
            "preflight": "compact",
            "strategies": [
                "native",
                "routersense_b_core_independent_async",
                "routersense_u_core_zero_raw_async",
            ],
        },
        "performance_smoke": {
            "warmup": 1,
            "measure": 2,
            "strategies": [
                "native",
                "routersense_b_core_independent_async",
                "routersense_u_core_zero_raw_async",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/closure/hotpath_cpu")
    parser.add_argument("--status", default="HOTPATH_CPU_CONTRACT_CLOSED")
    args = parser.parse_args()
    output_dir = REPO_ROOT / args.output_dir
    commit = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--short"]))
    perf_audit = _perf_profile_audit()
    manifest = _gpu_validation_manifest(commit=commit)
    status = {
        "status": str(args.status),
        "source_commit": commit,
        "git_dirty": dirty,
        "outputs": [
            "perf_profile_feature_audit.json",
            "gpu_validation_manifest.json",
            "status.json",
        ],
    }
    _write_json(output_dir / "perf_profile_feature_audit.json", perf_audit)
    _write_json(output_dir / "gpu_validation_manifest.json", manifest)
    _write_json(output_dir / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
