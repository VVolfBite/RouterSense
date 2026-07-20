"""Public runtime presets for the formal RouterSense strategy surface."""
from __future__ import annotations

import re
from typing import Any

PUBLIC_RUNTIME_LINES = {"phase_sync", "async_release"}
PUBLIC_OUTPUT_MODES = {"paper", "debug_replay"}
PUBLIC_STRATEGIES = {
    "disabled",
    "native",
    "fifo_async_p2p",
    "greedy_async_p2p",
    "birkhoff_phase_local_sync",
    "birkhoff_phase_local_async_p2p",
}
_AXIS_RE = re.compile(
    r"^routersense_(?P<timing>current|future)_"
    r"(?P<horizon>p012|p0123)_"
    r"(?P<scope>local|joint)_"
    r"(?P<engine>event|global)_"
    r"(?P<core>gmwd|rsbc|rscf)_async$"
)


def _axis_strategy(name: str) -> dict[str, Any] | None:
    match = _AXIS_RE.fullmatch(str(name))
    if match is None:
        return None
    timing, horizon = match.group("timing"), match.group("horizon")
    scope, engine, core = match.group("scope"), match.group("engine"), match.group("core")
    if timing == "future" and horizon != "p012":
        raise ValueError("future timing supports P012 only")
    prediction = timing == "future" or scope == "joint"
    return {
        "policy": "prepared_priority",
        "planner_id": f"{timing}:{horizon}:{scope}:{engine}:{core}",
        "run_kind": "online_policy_correctness",
        "execution_mode": "joint_window_async_p2p",
        "control_mode": "sync_before_phase",
        "p2_hint_mode": "calibrated_artifact" if prediction else "none",
        "calibrated_p2": bool(prediction),
        "online_p2_predictor": "copy_current_dispatch" if prediction else "none",
        "safe_projection_mode": "disabled",
        "planning_horizon": horizon,
        "planning_timing": "previous_layer" if timing == "future" else "on_demand",
        "p3_return_weight": 0.01 if horizon == "p0123" else 0.0,
        "planner_timing": timing,
        "planner_scope": scope,
        "planner_engine": engine,
        "planner_core": core,
    }


def is_public_strategy(strategy_name: str) -> bool:
    if str(strategy_name) in PUBLIC_STRATEGIES:
        return True
    try:
        return _axis_strategy(str(strategy_name)) is not None
    except ValueError:
        return False


def normalize_strategy_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"name": entry}
    if isinstance(entry, dict):
        return dict(entry)
    raise ValueError(f"unsupported strategy entry: {entry!r}")


def uses_public_runtime_surface(comparison: dict[str, Any]) -> bool:
    runtime = dict(comparison.get("runtime", {}) or {})
    return "line" in runtime or "output_mode" in runtime


def validate_public_runtime_surface(comparison: dict[str, Any]) -> None:
    runtime = dict(comparison.get("runtime", {}) or {})
    line = str(runtime.get("line", "phase_sync"))
    output_mode = str(runtime.get("output_mode", "paper"))
    if line not in PUBLIC_RUNTIME_LINES:
        raise ValueError(f"unsupported runtime.line {line!r}")
    if output_mode not in PUBLIC_OUTPUT_MODES:
        raise ValueError(f"unsupported runtime.output_mode {output_mode!r}")
    execution = dict(comparison.get("execution", {}) or {})
    if "bucket_rows" in execution:
        raise ValueError("public recommended configs must not expose execution.bucket_rows")
    if "control_mode" in runtime:
        raise ValueError("public recommended configs must not expose runtime.control_mode")
    observation = dict(comparison.get("observation", {}) or {})
    for key in {"profile", "capture_enabled", "heartbeat_enabled", "per_wave_timing_enabled", "replay_trace_enabled"}:
        if key in observation:
            raise ValueError(f"public recommended configs must not expose observation.{key}")
    for raw in comparison.get("strategies", []) or ():
        strategy = normalize_strategy_entry(raw)
        if not is_public_strategy(str(strategy.get("name", ""))):
            raise ValueError(f"unsupported public strategy {strategy.get('name')!r}")
        for key in ("execution_mode", "control_mode", "p2_hint_mode", "calibrated_p2", "policy"):
            if key in strategy:
                raise ValueError(f"public recommended configs must not expose strategy.{key}")


def public_runtime_defaults(*, output_mode: str) -> dict[str, Any]:
    if output_mode not in PUBLIC_OUTPUT_MODES:
        raise ValueError(f"unsupported output_mode {output_mode!r}")
    debug = output_mode == "debug_replay"
    return {
        "observation": {
            "profile": "debug" if debug else "perf",
            "capture_enabled": False,
            "capture_layer_selector": "",
            "capture_phase_selector": "",
            "heartbeat_enabled": debug,
            "per_wave_timing_enabled": debug,
            "replay_trace_enabled": debug,
        },
        "execution": {"bucket_mode": "dynamic_current", "bucket_rows": 0, "safe_projection_mode": "host_select"},
    }


def resolve_strategy_runtime(*, strategy_name: str, runtime_line: str) -> dict[str, Any]:
    if runtime_line not in PUBLIC_RUNTIME_LINES:
        raise ValueError(f"unsupported runtime_line {runtime_line!r}")
    formal = _axis_strategy(str(strategy_name))
    if formal is not None:
        if runtime_line != "async_release":
            raise ValueError("formal orthogonal strategies require runtime.line=async_release")
        return formal
    if strategy_name in {"disabled", "native"}:
        return {"policy": "", "run_kind": "online_observe", "execution_mode": "native_passthrough", "control_mode": "none", "p2_hint_mode": "none", "calibrated_p2": False, "online_p2_predictor": "none"}
    baselines = {
        "fifo_async_p2p": ("fifo_bucket", "joint_window_async_p2p"),
        "greedy_async_p2p": ("greedy_bucket", "joint_window_async_p2p"),
        "birkhoff_phase_local_sync": ("birkhoff_bucket_phase_local", "phase_sync_wave"),
        "birkhoff_phase_local_async_p2p": ("birkhoff_bucket_phase_local", "joint_window_async_p2p"),
    }
    if strategy_name in baselines:
        policy, mode = baselines[strategy_name]
        return {"policy": policy, "run_kind": "online_policy_correctness", "execution_mode": mode, "control_mode": "sync_before_phase", "p2_hint_mode": "none", "calibrated_p2": False, "online_p2_predictor": "none"}
    raise ValueError(f"unsupported strategy {strategy_name!r}")

__all__ = [
    "PUBLIC_OUTPUT_MODES", "PUBLIC_RUNTIME_LINES", "PUBLIC_STRATEGIES",
    "is_public_strategy", "normalize_strategy_entry", "public_runtime_defaults",
    "resolve_strategy_runtime", "uses_public_runtime_surface", "validate_public_runtime_surface",
]
