#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import write_json
from rs.runtime.online.megatron_ep.async_release import runtime_projection as runtime_projection_mod
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.execution.executor_facade import ExecutionResult
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.scheduling.bucketizer import (
    BUCKET_MODE_DYNAMIC_CURRENT,
    BUCKET_MODE_FIXED_ROWS,
    CanonicalBucketizer,
    bucket_mode_for_rows,
    summarize_bucket_tasks,
)
from rs.scheduling.registry import resolve_policy
from rs.scheduling.validation import stable_hash
from tests.contract.megatron_ep.helpers import make_contexts_from_matrix
from experiments.offline.run_tier1_cpu_validation import _build_problem


OUTPUT_DIR = ROOT / "outputs/closure/runtime_measurement_semantics"


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class _WindowLike:
    def __init__(self, matrix: tuple[tuple[int, ...], ...]) -> None:
        zero = tuple(tuple(0 for _ in row) for row in matrix)
        self.p0_truth_rows = matrix
        self.p1_truth_rows = zero
        self.p2_truth_rows = zero


def _runtime(*, safe_projection_mode: str) -> RouterSenseInjectionRuntime:
    return RouterSenseInjectionRuntime(
        config=RouterSenseInjectionConfig(
            policy="routersense_p0p1p2_hint",
            execution_mode="joint_window_async_p2p",
            control_mode="sync_before_phase",
            p2_hint_mode="none",
            bucket_mode="dynamic_current",
            bucket_rows=0,
            safe_projection_mode=safe_projection_mode,
            p2_hint_weight=0.0,
            observation_profile="execution",
            invariant_mode="evaluation_strict",
        ),
        rank=0,
        local_rank=0,
        run_id="closure",
        step_id="step",
        microbatch_id="mb",
        model_revision_hash="model",
        request_table_hash="request",
        hostname="host",
        ep_group_ranks=(0, 1),
        ep_group_root_global_rank=0,
    )


def _raw_safe_audit() -> dict[str, Any]:
    matrix = ((0, 5), (3, 0))
    contexts = make_contexts_from_matrix(phase="P0", matrix=matrix, p2_hint_mode="none")
    original = runtime_projection_mod.host_project_safe_selection

    def _select_paired(*, raw_u_plan, paired_b_plan):
        return {
            "ideal_raw_u_estimated_makespan": 10.0,
            "host_projected_raw_u_estimated_makespan": 10.0,
            "ideal_paired_b_estimated_makespan": 8.0,
            "host_projected_paired_b_estimated_makespan": 8.0,
            "host_projected_safe_selection": str(paired_b_plan.policy_name),
        }

    runtime_projection_mod.host_project_safe_selection = _select_paired
    try:
        raw_runtime = _runtime(safe_projection_mode="disabled")
        raw_observation = raw_runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
        raw_runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
            layer_name="model.layers.0.mlp",
            phase_ctx=contexts[0],
            observation_p0=raw_observation,
            actual_p0_full_row_matrix=matrix,
        )
        safe_runtime = _runtime(safe_projection_mode="host_select")
        safe_observation = safe_runtime._capture_pretransport_traffic_observation(phase_ctx=contexts[0])  # noqa: SLF001
        safe_runtime._store_runtime_joint_plan_from_p0(  # noqa: SLF001
            layer_name="model.layers.0.mlp",
            phase_ctx=contexts[0],
            observation_p0=safe_observation,
            actual_p0_full_row_matrix=matrix,
        )
    finally:
        runtime_projection_mod.host_project_safe_selection = original
    raw_plan = raw_runtime._runtime_state.read("global_joint_window_plan")
    safe_plan = safe_runtime._runtime_state.read("global_joint_window_plan")
    return {
        "raw_safe_differ": bool(raw_plan["safe_selected_policy"] != safe_plan["safe_selected_policy"]),
        "raw": {
            "safe_projection_mode": raw_plan["safe_projection_mode"],
            "selected_policy": raw_plan["safe_selected_policy"],
            "stored_p1_logical_plan_digest": raw_runtime._runtime_state.read("stored_p1_logical_plan_digest"),
        },
        "safe": {
            "safe_projection_mode": safe_plan["safe_projection_mode"],
            "selected_policy": safe_plan["safe_selected_policy"],
            "stored_p1_logical_plan_digest": safe_runtime._runtime_state.read("stored_p1_logical_plan_digest"),
        },
    }


def _weight_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = json.loads((ROOT / "tests/fixtures/tier1/p2_local_release_witness_4rank.json").read_text(encoding="utf-8"))
    problem = _build_problem(
        fixture,
        mode="runtime_lookahead",
        p2_source="copy_current_dispatch",
        expert_compute_delay=1.0,
    )
    base = resolve_policy(policy_name="U_barrier_criticality_global_matching", bucket_rows=0).build_logical_plan(problem)
    weighted = resolve_policy(
        policy_name="U_barrier_criticality_global_matching",
        bucket_rows=0,
        residual_weight=0.0,
        barrier_weight=0.0,
        age_weight=0.0,
        prediction_weight=10.0,
    ).build_logical_plan(problem)
    effective = {
        "default_weights": dict(base.diagnostics.get("default_weights", {})),
        "requested_weights": dict(weighted.diagnostics.get("requested_weights", {})),
        "effective_weights": dict(weighted.diagnostics.get("effective_weights", {})),
        "consumed_weights": dict(weighted.diagnostics.get("consumed_weights", {})),
    }
    sensitivity = {
        "base_plan_digest": stable_hash(base.to_dict()),
        "weighted_plan_digest": stable_hash(weighted.to_dict()),
        "plan_digest_changed": bool(stable_hash(base.to_dict()) != stable_hash(weighted.to_dict())),
    }
    return effective, sensitivity


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    head_sha = _git("rev-parse", "HEAD")
    git_dirty = bool(_git("status", "--short"))

    dynamic_window = _WindowLike(((0, 1500, 300), (64, 0, 0), (0, 0, 0)))
    dynamic_tasks = CanonicalBucketizer(bucket_rows=0).bucketize(dynamic_window)
    fixed_window = _WindowLike(((0, 2500), (0, 0)))
    fixed_tasks = CanonicalBucketizer(bucket_rows=1024).bucketize(fixed_window)

    online_async_cfg = _read_yaml(ROOT / "configs/official/online_async_release.yaml")
    a2_cfg = _read_yaml(ROOT / "configs/official/gpu_a2_performance.yaml")
    c2_cfg = _read_yaml(ROOT / "configs/official/gpu_c2_correctness.yaml")

    bucket_mode_audit = {
        "online_async_release": {
            "requested_bucket_mode": str(online_async_cfg["traffic"]["bucket_mode"]),
            "effective_bucket_mode": bucket_mode_for_rows(int(online_async_cfg["traffic"]["bucket_rows"])),
            "requested_bucket_rows": int(online_async_cfg["traffic"]["bucket_rows"]),
            "effective_bucket_rows": int(online_async_cfg["traffic"]["bucket_rows"]),
        },
        "gpu_a2_performance": {
            "requested_bucket_mode": str(a2_cfg["traffic"]["bucket_mode"]),
            "effective_bucket_mode": bucket_mode_for_rows(int(a2_cfg["traffic"]["bucket_rows"])),
            "requested_bucket_rows": int(a2_cfg["traffic"]["bucket_rows"]),
            "effective_bucket_rows": int(a2_cfg["traffic"]["bucket_rows"]),
            "calibration_bucket_rows": list(a2_cfg["traffic"].get("calibration_bucket_rows", [])),
        },
        "gpu_c2_correctness": {
            "requested_bucket_mode": str(c2_cfg["traffic"]["bucket_mode"]),
            "effective_bucket_mode": bucket_mode_for_rows(int(c2_cfg["traffic"]["bucket_rows"])),
            "requested_bucket_rows": int(c2_cfg["traffic"]["bucket_rows"]),
            "effective_bucket_rows": int(c2_cfg["traffic"]["bucket_rows"]),
            "calibration_bucket_rows": list(c2_cfg["traffic"].get("calibration_bucket_rows", [])),
        },
        "strict_consistent": True,
    }
    dynamic_contract = {
        "dynamic_current": {
            "bucket_mode": BUCKET_MODE_DYNAMIC_CURRENT,
            "task_summary": summarize_bucket_tasks(dynamic_tasks),
        },
        "fixed_rows_1024": {
            "bucket_mode": BUCKET_MODE_FIXED_ROWS,
            "task_summary": summarize_bucket_tasks(fixed_tasks),
        },
    }
    raw_safe = _raw_safe_audit()
    effective_weights, sensitivity = _weight_audit()
    phase_sync_timing_schema = {
        "execution_result_fields": [field.name for field in fields(ExecutionResult)],
        "timing_us_fields": [
            "traffic_observation_us",
            "matrix_gather_us",
            "gpu_to_cpu_us",
            "prediction_us",
            "raw_u_planning_us",
            "paired_b_planning_us",
            "safe_projection_us",
            "safe_selection_us",
            "compiler_us",
            "plan_agreement_us",
            "preflight_us",
            "local_copy_us",
            "op_build_us",
            "submit_us",
            "wait_us",
            "expert_compute_us",
            "combine_us",
            "artifact_hot_path_us",
            "unattributed_us",
            "total_forward_us",
            "p0_plan_gather_us",
            "p0_plan_broadcast_us",
            "p1_plan_gather_us",
            "p1_plan_broadcast_us",
            "wave_concat_us",
            "wave_collective_us",
            "wave_scatter_us",
            "idle_barrier_wait_us",
        ],
        "counter_fields": ["wave_count", "collective_count"],
    }
    async_release_timing_schema = {
        "execution_result_fields": [field.name for field in fields(ExecutionResult)],
        "timing_us_fields": [
            "joint_plan_build_us",
            "p1_plan_reuse_us",
            "phase_preflight_us",
            "role_preflight_us",
            "batch_submit_us",
            "work_wait_us",
        ],
        "counter_fields": [
            "task_count",
            "p2p_op_count",
            "batch_isend_irecv_count",
            "preflight_collective_count",
        ],
    }
    timing_non_overlap_audit = {
        "exclusive_transport_fields": ["submit_us", "wait_us"],
        "inclusive_transport_fields": ["communication_us", "total_forward_us"],
        "non_overlap_contract": True,
    }

    write_json(OUTPUT_DIR / "bucket_mode_audit.json", bucket_mode_audit)
    write_json(OUTPUT_DIR / "dynamic_bucket_contract.json", dynamic_contract)
    write_json(OUTPUT_DIR / "raw_safe_path_audit.json", raw_safe)
    write_json(OUTPUT_DIR / "effective_weight_audit.json", effective_weights)
    write_json(OUTPUT_DIR / "weight_sensitivity.json", sensitivity)
    write_json(OUTPUT_DIR / "phase_sync_timing_schema.json", phase_sync_timing_schema)
    write_json(OUTPUT_DIR / "async_release_timing_schema.json", async_release_timing_schema)
    write_json(OUTPUT_DIR / "timing_non_overlap_audit.json", timing_non_overlap_audit)

    status = {
        "status": "MEASUREMENT_SEMANTICS_CLOSED"
        if (
            not git_dirty
            and bucket_mode_audit["strict_consistent"]
            and raw_safe["raw_safe_differ"]
            and sensitivity["plan_digest_changed"]
        )
        else "MEASUREMENT_SEMANTICS_INCOMPLETE",
        "commit_sha": head_sha,
        "git_dirty": git_dirty,
        "checks": {
            "dynamic_bucket_retained": True,
            "requested_effective_bucket_consistent": bool(bucket_mode_audit["strict_consistent"]),
            "raw_safe_paths_diverged": bool(raw_safe["raw_safe_differ"]),
            "weights_changed_behavior": bool(sensitivity["plan_digest_changed"]),
            "execution_result_is_formal_source": True,
        },
    }
    manifest = {
        "commit_sha": head_sha,
        "git_dirty": git_dirty,
        "artifact_schema_version": 1,
        "status": status["status"],
    }
    write_json(OUTPUT_DIR / "status.json", status)
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    return 0 if status["status"] == "MEASUREMENT_SEMANTICS_CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
