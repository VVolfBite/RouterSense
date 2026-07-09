from __future__ import annotations

from typing import Any


PUBLIC_RUNTIME_LINES = {"phase_sync", "async_release"}
PUBLIC_OUTPUT_MODES = {"paper", "debug_replay"}
PUBLIC_STRATEGIES = {
    "disabled",
    "birkhoff_phase_local",
    "routersense_p0p1p2_hint",
    "routersense_joint_priority_phase_sync",
}


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
    if line == "async_release":
        raise ValueError("async_release runtime_line has a shadow-only skeleton but no online executor integration yet")
    execution = dict(comparison.get("execution", {}) or {})
    forbidden_execution = {"bucket_rows"}
    forbidden_runtime = {
        "control_mode",
    }
    forbidden_observation = {
        "profile",
        "capture_enabled",
        "heartbeat_enabled",
        "per_wave_timing_enabled",
        "replay_trace_enabled",
    }
    for key in forbidden_execution:
        if key in execution:
            raise ValueError(f"public recommended configs must not expose execution.{key}")
    for key in forbidden_runtime:
        if key in runtime:
            raise ValueError(f"public recommended configs must not expose runtime.{key}")
    observation = dict(comparison.get("observation", {}) or {})
    for key in forbidden_observation:
        if key in observation:
            raise ValueError(f"public recommended configs must not expose observation.{key}")
    for raw_strategy in comparison.get("strategies", []) or ():
        strategy = normalize_strategy_entry(raw_strategy)
        if str(strategy.get("name", "")) not in PUBLIC_STRATEGIES:
            raise ValueError(f"unsupported public strategy {strategy.get('name')!r}")
        for key in ("execution_mode", "control_mode", "p2_hint_mode", "calibrated_p2", "policy"):
            if key in strategy:
                raise ValueError(f"public recommended configs must not expose strategy.{key}")


def public_runtime_defaults(*, output_mode: str) -> dict[str, Any]:
    if output_mode == "paper":
        return {
            "observation": {
                "profile": "perf",
                "capture_enabled": False,
                "capture_layer_selector": "",
                "capture_phase_selector": "",
                "heartbeat_enabled": False,
                "per_wave_timing_enabled": False,
                "replay_trace_enabled": False,
            },
            "execution": {
                "bucket_rows": 0,
            },
        }
    if output_mode == "debug_replay":
        return {
            "observation": {
                "profile": "debug",
                "capture_enabled": False,
                "capture_layer_selector": "",
                "capture_phase_selector": "",
                "heartbeat_enabled": True,
                "per_wave_timing_enabled": True,
                "replay_trace_enabled": True,
            },
            "execution": {
                "bucket_rows": 0,
            },
        }
    raise ValueError(f"unsupported output_mode {output_mode!r}")


def resolve_strategy_runtime(*, strategy_name: str, runtime_line: str) -> dict[str, Any]:
    if runtime_line == "async_release":
        raise ValueError("async_release runtime_line has a shadow-only skeleton but no online executor integration yet")
    if strategy_name == "disabled":
        return {
            "policy": "",
            "run_kind": "online_observe",
            "execution_mode": "native_passthrough",
            "control_mode": "none",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
        }
    if strategy_name == "birkhoff_phase_local":
        return {
            "policy": "birkhoff_phase_local",
            "run_kind": "online_policy_correctness",
            "execution_mode": "phase_sync_wave",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
        }
    if strategy_name == "routersense_p0p1p2_hint":
        return {
            "policy": "routersense_p0p1p2_hint",
            "run_kind": "online_policy_correctness",
            "execution_mode": "multiphase_pending_window",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "calibrated_artifact",
            "calibrated_p2": True,
        }
    if strategy_name == "routersense_joint_priority_phase_sync":
        return {
            "policy": "routersense_joint_priority_phase_sync",
            "run_kind": "online_policy_correctness",
            "execution_mode": "phase_sync_wave",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "calibrated_artifact",
            "calibrated_p2": True,
        }
    raise ValueError(f"unsupported strategy {strategy_name!r}")
