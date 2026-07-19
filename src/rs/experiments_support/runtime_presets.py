from __future__ import annotations

import re
from typing import Any


PUBLIC_RUNTIME_LINES = {"phase_sync", "async_release"}
PUBLIC_OUTPUT_MODES = {"paper", "debug_replay"}
PUBLIC_STRATEGIES = {
    "disabled",
    "native",
    "birkhoff_phase_local",
    "birkhoff_phase_local_sync",
    "birkhoff_phase_local_async_p2p",
    "fifo_async_p2p",
    "greedy_async_p2p",
    "routersense_p0p1p2_hint",
    "routersense_joint_priority_phase_sync",
    "routersense_joint_phase_sync",
    "routersense_b_core_independent_async",
    "routersense_u_core_zero_raw_async",
    "routersense_u_core_predicted_raw_async",
    "routersense_u_core_predicted_safe_async",
    "routersense_joint_zero_raw_async",
    "routersense_joint_predicted_raw_async",
    "routersense_joint_zero_safe_async",
    "routersense_joint_predicted_safe_async",
    "routersense_joint_zero_hint_async_p2p",
    "routersense_joint_predicted_async_p2p",
    "routersense_safe_joint_async",
    "routersense_p012_async",
    "routersense_p0123_async",
    "routersense_future_p012_async",
}



_EXPLICIT_AXIS_STRATEGY_RE = re.compile(
    r"^routersense_(?P<timing>current|future)_"
    r"(?P<horizon>p012|p0123)_"
    r"(?P<scope>local|joint)_"
    r"(?P<engine>event|global)_"
    r"(?P<core>gmwd|rsbc|rscf)_async$"
)

_FORMAL_P012_STRATEGY_RE = re.compile(
    r"^routersense_(?P<horizon>p012|p0123|future_p012)_"
    r"(?P<branch>local|event|global)_(?P<core>gmwd|rsbc|rscf)_async$"
)


def _formal_p012_strategy(strategy_name: str) -> dict[str, Any] | None:
    """Resolve the public three-core P012 naming surface.

    Compatibility aliases without an explicit branch/core intentionally map to
    the current paper default: global RSCF.  The phase-policy ID remains the
    stable Megatron transport compatibility policy; ``planner_id`` selects the
    canonical formal planner used to construct the logical window plan.
    """

    explicit = _EXPLICIT_AXIS_STRATEGY_RE.fullmatch(str(strategy_name))
    if explicit is not None:
        timing = str(explicit.group("timing"))
        horizon = str(explicit.group("horizon"))
        scope = str(explicit.group("scope"))
        engine = str(explicit.group("engine"))
        core = str(explicit.group("core"))
        if timing == "future" and horizon != "p012":
            raise ValueError("future timing currently supports the P012 horizon only")
        uses_prediction = bool(timing == "future" or scope == "joint")
        return {
            "policy": "routersense_p0p1p2_hint",
            "planner_id": f"{timing}:{horizon}:{scope}:{engine}:{core}",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "calibrated_artifact" if uses_prediction else "none",
            "calibrated_p2": uses_prediction,
            "online_p2_predictor": "copy_current_dispatch" if uses_prediction else "none",
            "safe_projection_mode": "disabled",
            "planning_horizon": horizon,
            "planning_timing": "previous_layer" if timing == "future" else "on_demand",
            "p3_return_weight": 0.01 if horizon == "p0123" else 0.0,
            "planner_timing": timing,
            "planner_scope": scope,
            "planner_engine": engine,
            "planner_core": core,
        }

    aliases = {
        "routersense_p012_async": ("p012", "global", "rscf"),
        "routersense_p0123_async": ("p0123", "global", "rscf"),
        "routersense_future_p012_async": ("future_p012", "global", "rscf"),
    }
    parsed = aliases.get(str(strategy_name))
    if parsed is None:
        match = _FORMAL_P012_STRATEGY_RE.fullmatch(str(strategy_name))
        if match is None:
            return None
        parsed = (
            str(match.group("horizon")),
            str(match.group("branch")),
            str(match.group("core")),
        )
    horizon, branch, core = parsed
    if horizon in {"p0123", "future_p012"} and branch == "local":
        raise ValueError(f"{horizon} does not expose a local branch; use p012 local for the paired baseline")
    formal_prefix = "future_prepared" if horizon == "future_p012" else horizon
    planning_horizon = "p0123" if horizon == "p0123" else "p012"
    planning_timing = "previous_layer" if horizon == "future_p012" else "on_demand"
    return {
        "policy": "routersense_p0p1p2_hint",
        "planner_id": f"{formal_prefix}:{branch}:{core}",
        "run_kind": "online_policy_correctness",
        "execution_mode": "joint_window_async_p2p",
        "control_mode": "sync_before_phase",
        "p2_hint_mode": "calibrated_artifact",
        "calibrated_p2": True,
        "online_p2_predictor": "copy_current_dispatch",
        "safe_projection_mode": "disabled",
        "planning_horizon": planning_horizon,
        "planning_timing": planning_timing,
        "p3_return_weight": 0.01 if planning_horizon == "p0123" else 0.0,
        "planner_branch": branch,
        "planner_core": core,
    }


def is_public_strategy(strategy_name: str) -> bool:
    if str(strategy_name) in PUBLIC_STRATEGIES:
        return True
    try:
        return _formal_p012_strategy(str(strategy_name)) is not None
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
        if not is_public_strategy(str(strategy.get("name", ""))):
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
                "bucket_mode": "dynamic_current",
                "bucket_rows": 0,
                "safe_projection_mode": "host_select",
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
                "bucket_mode": "dynamic_current",
                "bucket_rows": 0,
                "safe_projection_mode": "host_select",
            },
        }
    raise ValueError(f"unsupported output_mode {output_mode!r}")


def resolve_strategy_runtime(*, strategy_name: str, runtime_line: str) -> dict[str, Any]:
    if runtime_line not in PUBLIC_RUNTIME_LINES:
        raise ValueError(f"unsupported runtime_line {runtime_line!r}")
    formal = _formal_p012_strategy(str(strategy_name))
    if formal is not None:
        if runtime_line != "async_release":
            raise ValueError("formal P012-family strategies require runtime.line=async_release")
        return formal
    if strategy_name in {"disabled", "native"}:
        return {
            "policy": "",
            "run_kind": "online_observe",
            "execution_mode": "native_passthrough",
            "control_mode": "none",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
        }
    if strategy_name == "birkhoff_phase_local":
        return {
            "policy": "birkhoff_phase_local",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p" if runtime_line == "async_release" else "phase_sync_wave",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
        }
    if strategy_name == "birkhoff_phase_local_sync":
        return {
            "policy": "birkhoff_phase_local",
            "run_kind": "online_policy_correctness",
            "execution_mode": "phase_sync_wave",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
        }
    if strategy_name == "birkhoff_phase_local_async_p2p":
        return {
            "policy": "birkhoff_phase_local",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
        }
    if strategy_name == "fifo_async_p2p":
        return {
            "policy": "bucketed_fifo",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
        }
    if strategy_name == "greedy_async_p2p":
        return {
            "policy": "greedy_ready_set",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
        }
    if strategy_name == "routersense_b_core_independent_async":
        return {
            "policy": "barrier_criticality_core_independent",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
            "safe_projection_mode": "disabled",
        }
    if strategy_name == "routersense_p0p1p2_hint":
        return {
            "policy": "routersense_p0p1p2_hint",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p" if runtime_line == "async_release" else "multiphase_pending_window",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "calibrated_artifact",
            "calibrated_p2": True,
            "online_p2_predictor": "copy_current_dispatch",
        }
    if strategy_name == "routersense_joint_priority_phase_sync":
        return {
            "policy": "routersense_joint_priority_phase_sync",
            "run_kind": "online_policy_correctness",
            "execution_mode": "phase_sync_wave",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "calibrated_artifact",
            "calibrated_p2": True,
            "online_p2_predictor": "copy_current_dispatch",
        }
    if strategy_name == "routersense_joint_phase_sync":
        return {
            "policy": "routersense_joint_priority_phase_sync",
            "run_kind": "online_policy_correctness",
            "execution_mode": "phase_sync_wave",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "calibrated_artifact",
            "calibrated_p2": True,
            "online_p2_predictor": "copy_current_dispatch",
        }
    if strategy_name == "routersense_joint_zero_hint_async_p2p":
        strategy_name = "routersense_joint_zero_raw_async"
    if strategy_name == "routersense_joint_predicted_async_p2p":
        strategy_name = "routersense_joint_predicted_raw_async"
    if strategy_name == "routersense_safe_joint_async":
        strategy_name = "routersense_joint_predicted_safe_async"
    if strategy_name == "routersense_joint_zero_raw_async":
        strategy_name = "routersense_u_core_zero_raw_async"
    if strategy_name == "routersense_u_core_zero_raw_async":
        return {
            "policy": "routersense_p0p1p2_hint",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
            "safe_projection_mode": "disabled",
        }
    if strategy_name == "routersense_joint_predicted_raw_async":
        strategy_name = "routersense_u_core_predicted_raw_async"
    if strategy_name == "routersense_u_core_predicted_raw_async":
        return {
            "policy": "routersense_p0p1p2_hint",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "calibrated_artifact",
            "calibrated_p2": True,
            "online_p2_predictor": "copy_current_dispatch",
            "safe_projection_mode": "disabled",
        }
    if strategy_name == "routersense_joint_zero_safe_async":
        return {
            "policy": "routersense_p0p1p2_hint",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "none",
            "calibrated_p2": False,
            "online_p2_predictor": "none",
            "safe_projection_mode": "host_select",
        }
    if strategy_name == "routersense_joint_predicted_safe_async":
        strategy_name = "routersense_u_core_predicted_safe_async"
    if strategy_name == "routersense_u_core_predicted_safe_async":
        return {
            "policy": "routersense_p0p1p2_hint",
            "run_kind": "online_policy_correctness",
            "execution_mode": "joint_window_async_p2p",
            "control_mode": "sync_before_phase",
            "p2_hint_mode": "calibrated_artifact",
            "calibrated_p2": True,
            "online_p2_predictor": "copy_current_dispatch",
            "safe_projection_mode": "host_select",
        }
    raise ValueError(f"unsupported strategy {strategy_name!r}")
