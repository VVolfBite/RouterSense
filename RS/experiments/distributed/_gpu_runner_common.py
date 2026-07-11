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


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for distributed GPU runners")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


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
) -> dict[str, Any]:
    model = dict(base_comparison.get("model", {}) or {})
    topology = dict(base_comparison.get("topology", {}) or {})
    runtime = dict(base_comparison.get("runtime", {}) or {})
    workload = dict(base_comparison.get("workload", {}) or {})
    execution = dict(base_comparison.get("execution", {}) or {})
    from experiments.online.support.runtime_presets import resolve_strategy_runtime

    strategy_runtime = resolve_strategy_runtime(strategy_name=strategy_name, runtime_line=str(runtime.get("line", "phase_sync")))
    is_native = not bool(strategy_runtime["policy"])
    p2_hint_mode = str(strategy_runtime["p2_hint_mode"])
    p2_hint_weight = 0.0 if p2_hint_mode == "none" else float(execution.get("p2_hint_weight", 1.0))
    return {
        "run": {"kind": str(strategy_runtime["run_kind"]), "name": run_name},
        "model": {
            "model_id": str(model.get("model_id", model.get("path", ""))),
            "local_path": str(model.get("path", model.get("local_path", ""))),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
        },
        "topology": {
            "launcher": {
                "kind": "torchrun",
                "nnodes": 1,
                "nproc_per_node": int(topology.get("ep_size", 1)),
                "standalone": True,
            },
            "ep": {"size": int(topology.get("ep_size", 1))},
            "network": {"scope": "single_node", "interface_hint": ""},
        },
        "workload": {"prompts": str(workload.get("prompts", "configs/workload/smoke_prompts.json"))},
        "runtime": {
            "precision": str(runtime.get("precision", "fp16")),
            "dispatcher": str(runtime.get("dispatcher", "alltoall")),
            "control_mode": "none" if is_native else str(strategy_runtime["control_mode"]),
        },
        "online_policy": {
            "name": "disabled" if is_native else str(strategy_runtime["policy"]),
            "parameters": {
                "p0_weight": float(execution.get("p0_weight", 1.0)),
                "p1_reservation_weight": float(execution.get("p1_reservation_weight", 1.0)),
                "p2_hint_weight": float(p2_hint_weight),
                "online_p2_predictor": str(strategy_runtime.get("online_p2_predictor", "none")),
            },
            "p2": {"mode": p2_hint_mode, "artifact": ""},
        },
        "offline_study": {"policies": []},
        "execution": {
            "mode": str(strategy_runtime["execution_mode"]),
            "bucket_rows": 0,
            "schedule": {
                "layer_selector": str(selected_layers),
                "phase_selector": str(execution.get("schedule_phase_selector", "both")),
            },
        },
        "observation": {
            "profile": str(profile),
            "capture_enabled": False,
            "capture_layer_selector": "",
            "capture_phase_selector": "",
            "heartbeat_enabled": profile == "debug",
            "per_wave_timing_enabled": profile != "perf",
            "replay_trace_enabled": profile == "debug",
        },
        "validation": {
            "save_logits": bool(save_logits),
            "stop_after_selected_layer": False,
        },
        "artifact": {"artifact_root": str(output_root)},
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
