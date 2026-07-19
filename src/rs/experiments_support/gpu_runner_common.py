from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rs.core.contracts.provenance import resolve_commit_identity
from rs.core.contracts.result import ONLINE_PIPELINE, RunIdentity
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for distributed GPU runners")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


def load_official_config(path: Path) -> dict[str, Any]:
    from rs.core.config_normalization import resolve_config_components

    return resolve_config_components(load_yaml(path), source_path=path)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for distributed GPU runners")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    pythonpath_entries = ["src", "."]
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    omp = env.get("OMP_NUM_THREADS", "").strip()
    if not omp or not omp.isdigit() or int(omp) <= 0:
        env["OMP_NUM_THREADS"] = "1"
    if not str(env.get("USE_LIBUV", "")).strip():
        env["USE_LIBUV"] = "0"
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        env.pop(key, None)
    return env


def run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = child_env()
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def copy_config(source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_dir / "input_config.yaml")


def build_policy_correctness_config(
    *,
    base_comparison: dict[str, Any],
    strategy_name: str,
    run_name: str,
    output_root: Path,
    profile: str,
    selected_layers: str,
    save_logits: bool,
    preflight_mode: str = "full",
    world_size: int | None = None,
) -> dict[str, Any]:
    from rs.core.layer_selection import resolve_layer_selector

    if str(preflight_mode) not in {"full", "compact"}:
        raise ValueError(f"unsupported preflight_mode: {preflight_mode!r}")
    model = dict(base_comparison.get("model", {}) or {})
    topology = dict(base_comparison.get("topology", {}) or {})
    topology_ep = dict(topology.get("ep", {}) or {})
    runtime = dict(base_comparison.get("runtime", {}) or {})
    traffic = dict(base_comparison.get("traffic", {}) or {})
    policy = dict(base_comparison.get("policy", {}) or {})
    policy_options = dict(policy.get("options", {}) or {})
    prediction = dict(base_comparison.get("prediction", {}) or {})
    workload = dict(base_comparison.get("workload", {}) or {})
    tokenization = dict(workload.get("tokenization", {}) or {})
    execution = dict(base_comparison.get("execution", {}) or {})
    selected_ep_size = int(
        world_size
        if world_size is not None
        else topology.get("ep_size", topology_ep.get("size", topology.get("world_size", 1))) or 1
    )
    from rs.experiments_support.runtime_presets import resolve_strategy_runtime

    strategy_runtime = resolve_strategy_runtime(strategy_name=strategy_name, runtime_line=str(runtime.get("line", "phase_sync")))
    is_native = not bool(strategy_runtime["policy"])
    p2_hint_mode = str(strategy_runtime["p2_hint_mode"])
    requested_invariant_mode = str(runtime.get("invariant_mode", "diagnostic"))
    requested_precision = str(runtime.get("precision", model.get("precision", "bf16")))
    requested_dispatcher = str(runtime.get("dispatcher", "alltoall"))
    effective_execution_mode = str(strategy_runtime["execution_mode"])
    if effective_execution_mode == "phase_sync_wave":
        effective_runtime_line = "phase_sync"
    elif effective_execution_mode == "native_passthrough":
        effective_runtime_line = str(runtime.get("line", "async_release"))
    else:
        effective_runtime_line = "async_release"
    requested_bucket_mode = str(traffic.get("bucket_mode", "dynamic_current"))
    requested_bucket_rows = int(traffic.get("bucket_rows", 0) or 0)
    selected_layer_ids = [
        str(item)
        for item in (
            ((base_comparison.get("evaluation", {}) or {}).get("selected_layer_ids"))
            or ((execution.get("schedule", {}) or {}).get("selected_layer_ids"))
            or ()
        )
    ]
    resolved_layer_selector = resolve_layer_selector(
        str(selected_layers),
        selected_layer_ids=selected_layer_ids,
        invariant_mode=requested_invariant_mode,
    )
    effective_safe_projection_mode = str(
        strategy_runtime.get(
            "safe_projection_mode",
            policy_options.get("safe_projection_mode", execution.get("safe_projection_mode", "host_select")),
        )
    )
    requested_p0_weight = float(policy_options.get("p0_weight", execution.get("p0_weight", 1.0)))
    requested_p1_weight = float(policy_options.get("p1_reservation_weight", execution.get("p1_reservation_weight", 1.0)))
    p2_hint_weight = 0.0 if p2_hint_mode == "none" else float(policy_options.get("p2_hint_weight", execution.get("p2_hint_weight", 1.0)))
    requested_residual_weight = float(policy_options.get("residual_weight", execution.get("residual_weight", 0.75)))
    requested_barrier_weight = float(policy_options.get("barrier_weight", execution.get("barrier_weight", 1.75)))
    requested_age_weight = float(policy_options.get("age_weight", execution.get("age_weight", 0.15)))
    requested_prediction_weight = float(policy_options.get("prediction_weight", execution.get("prediction_weight", 0.35)))
    requested_p3_return_weight = float(strategy_runtime.get("p3_return_weight", policy_options.get("p3_return_weight", execution.get("p3_return_weight", 0.0))))
    requested_planning_horizon = str(strategy_runtime.get("planning_horizon", policy_options.get("planning_horizon", "p012")))
    requested_planning_timing = str(strategy_runtime.get("planning_timing", policy_options.get("planning_timing", "legacy_auto")))
    requested_planner_id = str(strategy_runtime.get("planner_id", policy_options.get("planner_id", "")))
    requested_online_predictor = str(
        prediction.get(
            "name",
            strategy_runtime.get("online_p2_predictor", policy_options.get("online_p2_predictor", "none")),
        )
    )
    requested_online_predictor_config = dict(
        prediction.get("config", policy_options.get("online_p2_predictor_config", {})) or {}
    )
    model_local_path = str(model.get("local_path", model.get("path", "")))
    model_id = str(model.get("model_id", model_local_path))
    return {
        "run": {"kind": str(strategy_runtime["run_kind"]), "name": run_name},
        "model": {
            "model_id": model_id,
            "local_path": model_local_path,
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
        },
        "topology": {
            "launcher": {
                "kind": "torchrun",
                "nnodes": 1,
                "nproc_per_node": selected_ep_size,
                "standalone": True,
            },
            "ep": {"size": selected_ep_size},
            "network": {"scope": "single_node", "interface_hint": ""},
        },
        "workload": {
            "prompts": str(workload.get("prompts", "configs/workload/smoke_prompts.json")),
            "tokenization": {
                "padding": str(tokenization.get("padding", "longest")),
                "truncation": bool(tokenization.get("truncation", False)),
                "max_length": tokenization.get("max_length"),
                "expected_prompt_count": tokenization.get("expected_prompt_count"),
                "expected_batch_rows": tokenization.get("expected_batch_rows"),
                "expected_seq_len": tokenization.get("expected_seq_len"),
            },
        },
        "runtime": {
            "line": effective_runtime_line,
            "precision": requested_precision,
            "invariant_mode": requested_invariant_mode,
            "dispatcher": requested_dispatcher,
            "control_mode": "none" if is_native else str(strategy_runtime["control_mode"]),
        },
        "online_policy": {
            "name": "disabled" if is_native else str(strategy_runtime["policy"]),
            "parameters": {
                "planner_id": requested_planner_id,
                "p0_weight": requested_p0_weight,
                "p1_reservation_weight": requested_p1_weight,
                "p2_hint_weight": float(p2_hint_weight),
                "residual_weight": requested_residual_weight,
                "barrier_weight": requested_barrier_weight,
                "age_weight": requested_age_weight,
                "prediction_weight": requested_prediction_weight,
                "p3_return_weight": requested_p3_return_weight,
                "planning_horizon": requested_planning_horizon,
                "planning_timing": requested_planning_timing,
                "online_p2_predictor": requested_online_predictor,
                "online_p2_predictor_config": requested_online_predictor_config,
            },
            "p2": {"mode": p2_hint_mode, "artifact": ""},
        },
        "offline_study": {"policies": []},
        "execution": {
            "mode": effective_execution_mode,
            "bucket_mode": requested_bucket_mode,
            "bucket_rows": requested_bucket_rows,
            "safe_projection_mode": effective_safe_projection_mode,
            "preflight_mode": str(preflight_mode),
            "schedule": {
                "layer_selector": str(selected_layers),
                "phase_selector": str(execution.get("schedule_phase_selector", "both")),
                "selected_layer_ids": list(resolved_layer_selector.resolved_layer_ids),
            },
        },
        "observation": {
            "profile": str(profile),
            "capture_enabled": False,
            "capture_layer_selector": "",
            "capture_phase_selector": "",
            "heartbeat_enabled": profile == "debug",
            "per_wave_timing_enabled": profile not in {"perf", "timeline_light", "attribution_light"},
            "replay_trace_enabled": profile == "debug",
        },
        "validation": {
            "save_logits": bool(save_logits),
            "stop_after_selected_layer": False,
        },
        "artifact": {"artifact_root": str(output_root)},
        "evaluation": {
            "selected_layer_ids": list(resolved_layer_selector.resolved_layer_ids),
        },
        "requested_layer_selector": str(selected_layers),
        "resolved_layer_selector": str(resolved_layer_selector.resolved_selector),
        "resolved_layer_ids": list(resolved_layer_selector.resolved_layer_ids),
        "requested_preflight_mode": str(preflight_mode),
        "effective_preflight_mode": str(preflight_mode),
    }


def build_strategy_comparison_config(
    *,
    base_comparison: dict[str, Any],
    strategies: list[str],
    repetitions: int,
    profile: str,
    output_mode: str,
) -> dict[str, Any]:
    payload = json.loads(json.dumps(base_comparison))
    payload.setdefault("runtime", {})
    payload["runtime"]["line"] = "phase_sync"
    payload["runtime"]["output_mode"] = output_mode
    payload.setdefault("observation", {})
    payload["observation"]["profile"] = profile
    payload.setdefault("strategies", [])
    payload["strategies"] = [{"name": item} for item in strategies]
    payload.setdefault("execution", {})
    payload["execution"]["repetitions"] = int(repetitions)
    return payload


def torchrun_policy_command(*, config_path: Path, run_id: str, output_dir: Path, world_size: int, native: bool) -> list[str]:
    module = "experiments.online.collect_native_ep_trace" if native else "experiments.online.run_policy_correctness"
    if int(world_size) <= 1:
        return [
            sys.executable,
            "-m",
            module,
            "--config",
            str(config_path),
            "--run-id",
            run_id,
            "--output-dir",
            str(output_dir),
        ]
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(world_size)}",
        "-m",
        module,
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
    ]


def python_module_command(*, module: str, args: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *args]


def available_cuda_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_runner_result_bundle(
    output_dir: Path,
    *,
    runner_name: str,
    payload: dict[str, Any],
    run_kind: str,
    claim_scope: str = "diagnostic",
) -> None:
    commit_sha, git_dirty, _source = resolve_commit_identity(REPO_ROOT)
    status = str(payload.get("status", "")).strip()
    executed = status in {"passed", "executed"}
    completed = status == "passed"
    reserved_detail_keys = {
        "schema_version",
        "run_identity",
        "status",
        "correctness_status",
        "performance_status",
        "pipeline",
        "commit_sha",
        "git_clean",
        "instrumentation_mode",
        "audit_evidence_level",
        "measurement_complete",
        "eligibility",
        "summary",
        "details",
        "extensions",
    }
    payload_details = {
        str(key): value
        for key, value in dict(payload).items()
        if str(key) not in reserved_detail_keys
    }
    summary = {
        "run_kind": str(run_kind),
        "all_work_completed": bool(completed),
        "fallback_count": int(payload.get("fallback_count", 0) or 0),
        "timeout_count": int(payload.get("timeout_count", 0) or 0),
        "check_failure_count": int(payload.get("check_failure_count", 0) or 0),
        "cleanup_failure_count": int(payload.get("cleanup_failure_count", 0) or 0),
        "execution_outcome_count": 1 if executed else 0,
        "missing_execution_outcome_count": 0 if executed else 1,
        "formal_execution_expected": False,
        "runner_name": str(runner_name),
        "runner_status": status,
        "performance_measurement_complete": False,
        "measured_repeat_count": 0,
        "warmup_excluded": False,
        "preparation_miss_count": int(payload.get("preparation_miss_count", 0) or 0),
        "provisional_execution_count": int(payload.get("provisional_execution_count", 0) or 0),
        "materialization_failure_count": int(payload.get("materialization_failure_count", 0) or 0),
        "execution_failure_count": int(payload.get("execution_failure_count", 0) or 0),
        "native_fallback_count": int(payload.get("native_fallback_count", 0) or 0),
        "semantic_failure_fallback_count": int(payload.get("semantic_failure_fallback_count", 0) or 0),
        "safe_selector_fallback_count": int(payload.get("safe_selector_fallback_count", 0) or 0),
    }
    details = {
        "run_kind": str(run_kind),
        "runner_name": str(runner_name),
        "runner_status": status,
        "generated_at": _utc_now(),
        **payload_details,
    }
    bundle = build_result_bundle(
        ResultBundleDraft(
            run_identity=RunIdentity(
                run_id=str(output_dir.resolve().name),
                pipeline=ONLINE_PIPELINE,
                claim_scope=str(claim_scope),
                trace_origin="derived_runner",
                future_information_mode="predicted",
            ),
            status="success" if executed else "invalid",
            correctness_status="valid" if completed else "invalid",
            performance_status="ineligible",
            commit_sha=str(commit_sha or "unknown"),
            git_clean=bool(not git_dirty),
            instrumentation_mode="contract",
            audit_evidence_level="summary_only",
            measurement_complete=False,
            summary=summary,
            details=details,
            extensions={},
        )
    )
    write_json(output_dir / "result_bundle.json", bundle.to_dict())
