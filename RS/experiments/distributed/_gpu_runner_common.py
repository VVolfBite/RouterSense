from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


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
    env["PYTHONPATH"] = "src:." if not existing else f"src:.:{existing}"
    omp = env.get("OMP_NUM_THREADS", "").strip()
    if not omp or not omp.isdigit() or int(omp) <= 0:
        env["OMP_NUM_THREADS"] = "1"
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
    selected_ep_size = int(topology.get("ep_size", topology_ep.get("size", topology.get("world_size", 1))) or 1)
    from experiments.online.support.runtime_presets import resolve_strategy_runtime

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
    requested_online_predictor = str(
        strategy_runtime.get(
            "online_p2_predictor",
            prediction.get("name", policy_options.get("online_p2_predictor", "none")),
        )
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
                "p0_weight": requested_p0_weight,
                "p1_reservation_weight": requested_p1_weight,
                "p2_hint_weight": float(p2_hint_weight),
                "residual_weight": requested_residual_weight,
                "barrier_weight": requested_barrier_weight,
                "age_weight": requested_age_weight,
                "prediction_weight": requested_prediction_weight,
                "online_p2_predictor": requested_online_predictor,
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
