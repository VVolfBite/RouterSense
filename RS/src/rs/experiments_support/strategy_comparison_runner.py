from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from rs.core.contracts.provenance import resolve_commit_identity
from rs.core.contracts.result import ONLINE_PIPELINE, ResultBundle, RunIdentity
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle
from rs.reporting.comparison_metrics import (
    aggregate_repetitions,
    build_comparison_report,
    metrics_from_rank_dir,
    render_markdown_report,
)
from rs.reporting.prepared_plan_runtime_analysis import analyze_prepared_plan_runtime
from rs.runtime.online.megatron_ep.observation import write_json

from .runtime_presets import (
    normalize_strategy_entry,
    public_runtime_defaults,
    resolve_strategy_runtime,
    uses_public_runtime_surface,
    validate_public_runtime_surface,
)

ROOT = Path(__file__).resolve().parents[3]


def _canonical_online_to_runtime_view(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(payload.get("runtime", {}) or {})
    traffic = dict(payload.get("traffic", {}) or {})
    policy = dict(payload.get("policy", {}) or {})
    evaluation = dict(payload.get("evaluation", {}) or {})
    return {
        "schema_version": 1,
        "run": dict(payload.get("run", {}) or {}),
        "model": dict(payload.get("model", {}) or {}),
        "topology": dict(payload.get("topology", {}) or {}),
        "workload": dict(payload.get("workload", {}) or {}),
        "runtime": {
            "line": str(runtime.get("line", "phase_sync")),
            "output_mode": str(runtime.get("output_mode", "paper")),
            "precision": str(runtime.get("precision", "fp16")),
            "dispatcher": str(runtime.get("dispatcher", "alltoall")),
            "invariant_mode": str(runtime.get("invariant_mode", "diagnostic")),
            "selected_layers": str(runtime.get("selected_layers", "all")),
        },
        "strategies": [dict(item) for item in payload.get("strategies", ()) or ()],
        "execution": {
            "repetitions": int(evaluation.get("repeats", 1) or 1),
            "warmup": int(evaluation.get("warmup", 0) or 0),
            "bucket_mode": str(traffic.get("bucket_mode", "dynamic_current")),
            "bucket_rows": int(traffic.get("bucket_rows", 0) or 0),
            "safe_projection_mode": str((policy.get("options", {}) or {}).get("safe_projection_mode", "host_select")),
            "p0_weight": float((policy.get("options", {}) or {}).get("p0_weight", 1.0)),
            "p1_reservation_weight": float((policy.get("options", {}) or {}).get("p1_reservation_weight", 1.0)),
            "p2_hint_weight": float((policy.get("options", {}) or {}).get("p2_hint_weight", 1.0)),
            "residual_weight": float((policy.get("options", {}) or {}).get("residual_weight", 0.75)),
            "barrier_weight": float((policy.get("options", {}) or {}).get("barrier_weight", 1.75)),
            "age_weight": float((policy.get("options", {}) or {}).get("age_weight", 0.15)),
            "prediction_weight": float((policy.get("options", {}) or {}).get("prediction_weight", 0.35)),
            "schedule_layer_selector": str(runtime.get("selected_layers", "all")),
            "schedule_phase_selector": str(evaluation.get("phase_selector", "both")),
        },
        "comparison": {
            "baseline_strategy": str(evaluation.get("baseline_strategy", "")),
            "metrics": list(evaluation.get("metrics", ()) or ()),
        },
        "_canonical_public_entry": True,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"comparison config must be a mapping: {path}")
    if int(payload.get("schema_version", 0) or 0) == 1 and str((payload.get("run", {}) or {}).get("kind", "")) == "online_strategy_comparison":
        return _canonical_online_to_runtime_view(payload)
    return payload


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def model_config(model: dict[str, Any], output_dir: Path) -> str:
    path = output_dir / "generated_configs" / "model.yaml"
    payload = {
        "model_id": str(model.get("model_id", model.get("path", ""))),
        "local_path": str(model.get("path", model.get("local_path", ""))),
        "trust_remote_code": bool(model.get("trust_remote_code", False)),
    }
    dump_yaml(path, payload)
    return str(path)


def topology_config(topology: dict[str, Any], output_dir: Path) -> str:
    ep_size = int(topology.get("ep_size", 1))
    path = output_dir / "generated_configs" / "topology.yaml"
    payload = {
        "launcher": {"kind": "torchrun", "nnodes": 1, "nproc_per_node": ep_size, "standalone": True},
        "ep": {"size": ep_size},
        "network": {"scope": "single_node", "interface_hint": ""},
    }
    dump_yaml(path, payload)
    return str(path)


def single_strategy_config(
    *,
    comparison: dict[str, Any],
    strategy: dict[str, Any],
    repetition: int,
    output_dir: Path,
    model_config_path: str,
    topology_config_path: str,
) -> Path:
    strategy = normalize_strategy_entry(strategy)
    execution = comparison.get("execution", {}) or {}
    runtime = comparison.get("runtime", {}) or {}
    workload = comparison.get("workload", {}) or {}
    observation = comparison.get("observation", {}) or {}
    validation = comparison.get("validation", {}) or {}
    run_name = f"rep{repetition}"
    public_surface = uses_public_runtime_surface(comparison)
    if public_surface:
        line = str(runtime.get("line", "phase_sync"))
        output_mode = str(runtime.get("output_mode", "paper"))
        runtime_defaults = public_runtime_defaults(output_mode=output_mode)
        strategy_runtime = resolve_strategy_runtime(strategy_name=str(strategy.get("name", "")), runtime_line=line)
        policy_name = str(strategy_runtime["policy"])
        is_native_baseline = not policy_name
        p2_mode = str(strategy_runtime["p2_hint_mode"])
        calibrated_p2 = bool(strategy_runtime["calibrated_p2"])
        online_p2_predictor = str(strategy_runtime.get("online_p2_predictor", "copy_current_dispatch"))
        control_mode = str(strategy_runtime["control_mode"])
        execution_mode = str(strategy_runtime["execution_mode"])
        run_kind = str(strategy_runtime["run_kind"])
        safe_projection_mode = str(strategy_runtime.get("safe_projection_mode", runtime_defaults["execution"].get("safe_projection_mode", "host_select")))
        effective_observation = dict(runtime_defaults["observation"])
        effective_observation["invariant_mode"] = str(runtime.get("invariant_mode", "diagnostic"))
        effective_execution_bucket_mode = str(runtime_defaults["execution"].get("bucket_mode", "dynamic_current"))
        effective_execution_bucket_rows = int(runtime_defaults["execution"]["bucket_rows"])
    else:
        policy_name = str(strategy.get("policy", ""))
        is_native_baseline = not policy_name
        p2_mode = str(strategy.get("p2_hint_mode", "none"))
        calibrated_p2 = bool(strategy.get("calibrated_p2", False))
        online_p2_predictor = str(strategy.get("online_p2_predictor", "copy_current_dispatch"))
        safe_projection_mode = str(strategy.get("safe_projection_mode", execution.get("safe_projection_mode", "host_select")))
        control_mode = "none" if is_native_baseline else str(strategy.get("control_mode", "sync_before_phase"))
        execution_mode = "native_passthrough" if is_native_baseline else str(strategy.get("execution_mode", "phase_sync_wave"))
        run_kind = "online_observe" if is_native_baseline else "online_policy_correctness"
        effective_observation = {
            "profile": str(observation.get("profile", "perf")),
            "capture_enabled": bool(observation.get("capture_enabled", False)),
            "capture_layer_selector": str(observation.get("capture_layer_selector", "")),
            "capture_phase_selector": str(observation.get("capture_phase_selector", "")),
            "heartbeat_enabled": bool(observation.get("heartbeat_enabled", False)),
            "per_wave_timing_enabled": bool(observation.get("per_wave_timing_enabled", False)),
            "replay_trace_enabled": bool(observation.get("replay_trace_enabled", False)),
            "invariant_mode": str(observation.get("invariant_mode", runtime.get("invariant_mode", "diagnostic"))),
        }
        effective_execution_bucket_mode = str(execution.get("bucket_mode", "dynamic_current"))
        effective_execution_bucket_rows = 0 if is_native_baseline else int(execution.get("bucket_rows", 0))
    p2_hint_weight = float(execution.get("p2_hint_weight", 1.0))
    if p2_mode in {"none", "deterministic_stub"} and not calibrated_p2:
        p2_hint_weight = 0.0
    config = {
        "run": {"kind": run_kind, "name": run_name},
        "model": {"config": model_config_path},
        "topology": {"config": topology_config_path},
        "workload": {"prompts": str(workload.get("prompts", "configs/workload/smoke_prompts.json"))},
        "runtime": {
            "precision": str(runtime.get("precision", "fp16")),
            "dispatcher": str(runtime.get("dispatcher", "alltoall")),
            "control_mode": control_mode,
        },
        "online_policy": {
            "name": "disabled" if is_native_baseline else policy_name,
            "parameters": {
                "p0_weight": float(execution.get("p0_weight", 1.0)),
                "p1_reservation_weight": float(execution.get("p1_reservation_weight", 1.0)),
                "p2_hint_weight": p2_hint_weight,
                "residual_weight": float(execution.get("residual_weight", 0.75)),
                "barrier_weight": float(execution.get("barrier_weight", 1.75)),
                "age_weight": float(execution.get("age_weight", 0.15)),
                "prediction_weight": float(execution.get("prediction_weight", 0.35)),
                "online_p2_predictor": online_p2_predictor,
            },
            "p2": {"mode": p2_mode, "artifact": ""},
        },
        "offline_study": {"policies": []},
        "execution": {
            "mode": execution_mode,
            "bucket_mode": effective_execution_bucket_mode,
            "bucket_rows": effective_execution_bucket_rows,
            "safe_projection_mode": safe_projection_mode,
            "schedule": {
                "layer_selector": str(execution.get("schedule_layer_selector", "all")),
                "phase_selector": str(execution.get("schedule_phase_selector", "both")),
            },
        },
        "observation": effective_observation,
        "validation": {
            "save_logits": bool(validation.get("save_logits", False)),
            "stop_after_selected_layer": bool(validation.get("stop_after_selected_layer", False)),
        },
        "artifact": {"artifact_root": str(output_dir / "per_strategy" / str(strategy["name"]))},
    }
    if calibrated_p2:
        config["online_policy"]["p2"]["mode"] = "calibrated_artifact"
    target = output_dir / "generated_configs" / f"{strategy['name']}_rep{repetition}.yaml"
    dump_yaml(target, config)
    return target


def strategy_run_kind(*, comparison: dict[str, Any], strategy: dict[str, Any]) -> str:
    strategy = normalize_strategy_entry(strategy)
    if uses_public_runtime_surface(comparison):
        runtime = comparison.get("runtime", {}) or {}
        strategy_runtime = resolve_strategy_runtime(
            strategy_name=str(strategy.get("name", "")),
            runtime_line=str(runtime.get("line", "phase_sync")),
        )
        return str(strategy_runtime["run_kind"])
    policy_name = str(strategy.get("policy", ""))
    if not policy_name:
        return "online_observe"
    return "online_policy_correctness"


def entrypoint_module(*, run_kind: str) -> str:
    if run_kind == "online_observe":
        return "experiments.online.collect_native_ep_trace"
    return "experiments.online.run_policy_correctness"


def torchrun_command(*, ep_size: int, config_path: Path, run_id: str, strategy_dir: Path, run_kind: str) -> list[str]:
    if int(ep_size) == 1:
        return [
            os.fspath(Path(os.sys.executable).resolve()),
            "-m",
            entrypoint_module(run_kind=run_kind),
            "--config",
            str(config_path),
            "--run-id",
            run_id,
            "--output-dir",
            str(strategy_dir),
        ]
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={ep_size}",
        "-m",
        entrypoint_module(run_kind=run_kind),
        "--config",
        str(config_path),
        "--run-id",
        run_id,
        "--output-dir",
        str(strategy_dir),
    ]


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        env.pop(key, None)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in ("src", existing) if part)
    omp = env.get("OMP_NUM_THREADS", "").strip()
    if not omp or not omp.isdigit() or int(omp) <= 0:
        env["OMP_NUM_THREADS"] = "1"
    return env


def copy_config(source: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_dir / "config.yaml")


def rank0_orchestrates_only() -> bool:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return True
    return str(local_rank) == "0"


def write_runtime_analysis(run_dir: Path) -> None:
    report = analyze_prepared_plan_runtime(run_dir, rank=0)
    write_json(run_dir / "prepared_plan_runtime_analysis.json", report)


def read_summary(run_dir: Path) -> dict[str, Any]:
    bundle_path = run_dir / "result_bundle.json"
    if bundle_path.exists():
        try:
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle = ResultBundle.from_dict(payload)
            details = dict(bundle.details)
            if details:
                return details
            return dict(bundle.summary)
        except Exception:
            pass
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    details = payload.get("details", {}) if isinstance(payload, dict) else {}
    if isinstance(details, dict):
        return details
    return {}


def write_result_bundle(output_dir: Path, *, report: dict[str, Any], timing: dict[str, Any], dry_run: bool) -> None:
    commit_sha, git_dirty, _source = resolve_commit_identity(ROOT)
    bundle = build_result_bundle(
        ResultBundleDraft(
            run_identity=RunIdentity(
                run_id=str(output_dir.resolve().name),
                pipeline=ONLINE_PIPELINE,
                claim_scope="diagnostic",
                trace_origin="derived_runner",
                future_information_mode="predicted",
            ),
            status="success",
            correctness_status="invalid" if dry_run else "valid",
            performance_status="ineligible",
            commit_sha=str(commit_sha or "unknown"),
            git_clean=bool(not git_dirty),
            instrumentation_mode="contract",
            audit_evidence_level="summary_only",
            measurement_complete=False,
            summary={
                "run_kind": "ONLINE_COMPARISON",
                "all_work_completed": bool(not dry_run),
                "fallback_count": 0,
                "timeout_count": 0,
                "check_failure_count": 0,
                "cleanup_failure_count": 0,
                "execution_outcome_count": 0 if dry_run else 1,
                "missing_execution_outcome_count": 1 if dry_run else 0,
                "formal_execution_expected": False,
                "strategy_count": int(len(report.get("strategies", ()))),
                "comparison_report_generated": True,
            },
            details={
                "run_kind": "ONLINE_COMPARISON",
                "comparison_report": report,
                "timing": timing,
                "dry_run": bool(dry_run),
            },
            extensions={},
        )
    )
    write_json(output_dir / "result_bundle.json", bundle.to_dict())


def run_strategy_comparison(*, config_path: Path, output_dir: Path, dry_run: bool) -> int:
    if not rank0_orchestrates_only():
        return 0
    comparison = load_yaml(config_path)
    if uses_public_runtime_surface(comparison) and not bool(comparison.get("_normalized_public_bridge", False)):
        validate_public_runtime_surface(comparison)
    copy_config(config_path, output_dir)
    model_config_path = model_config(comparison.get("model", {}) or {}, output_dir)
    topology_config_path = topology_config(comparison.get("topology", {}) or {}, output_dir)
    strategies = [normalize_strategy_entry(item) for item in list(comparison.get("strategies", []) or [])]
    repetitions = int((comparison.get("execution", {}) or {}).get("repetitions", 1))
    ep_size = int((comparison.get("topology", {}) or {}).get("ep_size", 1))
    baseline = str((comparison.get("comparison", {}) or {}).get("baseline_strategy", strategies[0]["name"] if strategies else ""))
    timing: dict[str, Any] = {}
    strategy_entries: list[dict[str, Any]] = []
    for strategy in strategies:
        name = str(strategy["name"])
        strategy_dir = output_dir / "per_strategy" / name
        repetition_metrics = []
        timing[name] = []
        for repetition in range(repetitions):
            run_id = f"rep{repetition}"
            run_dir = strategy_dir / run_id
            run_kind = strategy_run_kind(comparison=comparison, strategy=strategy)
            generated = single_strategy_config(
                comparison=comparison,
                strategy=strategy,
                repetition=repetition,
                output_dir=output_dir,
                model_config_path=model_config_path,
                topology_config_path=topology_config_path,
            )
            cmd = torchrun_command(
                ep_size=ep_size,
                config_path=generated,
                run_id=run_id,
                strategy_dir=strategy_dir,
                run_kind=run_kind,
            )
            timing_start = time.monotonic_ns()
            if dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "command.txt").write_text(" ".join(cmd), encoding="utf-8")
                return_code = 0
            else:
                proc = subprocess.run(cmd, cwd=ROOT, env=child_env(), check=False)
                return_code = int(proc.returncode)
            elapsed_us = (time.monotonic_ns() - timing_start) / 1000.0
            timing[name].append({"repetition": repetition, "elapsed_us": elapsed_us, "return_code": return_code})
            if return_code != 0:
                raise SystemExit(return_code)
            if not dry_run:
                write_runtime_analysis(run_dir)
                metrics = metrics_from_rank_dir(run_dir, rank=0)
                metrics["total_forward_us"] = elapsed_us
                repetition_metrics.append(metrics)
        aggregated = aggregate_repetitions(repetition_metrics) if repetition_metrics else {}
        strategy_entries.append(
            {
                "name": name,
                "family": str(strategy.get("family", "")),
                "description": str(strategy.get("description", "")),
                "repetitions": repetitions,
                "metrics": aggregated,
            }
        )
    report = build_comparison_report(run_id=output_dir.name, baseline=baseline, strategies=strategy_entries)
    write_json(output_dir / "timing.json", timing)
    write_json(output_dir / "comparison_report.json", report)
    write_result_bundle(output_dir, report=report, timing=timing, dry_run=bool(dry_run))
    (output_dir / "comparison_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return 0


__all__ = ["run_strategy_comparison"]
