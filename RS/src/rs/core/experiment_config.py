"""Typed experiment configuration for formal offline and online entrypoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rs.core.contracts.observation import RuntimeObservationConfig


SUPPORTED_RUN_KINDS = {
    "offline_trace",
    "offline_flow_study",
    "online_observe",
    "online_policy_correctness",
}

SUPPORTED_OBSERVATION_PROFILES = {"minimal", "perf", "timeline_light", "attribution_light", "execution", "debug"}
SUPPORTED_CONTROL_MODES = {"none", "default_continue", "sync_before_phase"}
SUPPORTED_EXECUTION_MODES = {"native_passthrough", "phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"}
SUPPORTED_ONLINE_PHASE_POLICIES = {
    "phase_barrier_fifo",
    "bucketed_fifo",
    "greedy_ready_set",
    "islip_round_robin",
    "power_of_two_choices",
    "birkhoff_phase_local",
    "trivial_reverse_bucket",
    "aurora_order_fixed",
    "fast_bvn_single_tier",
    "barrier_criticality_core_independent",
    "routersense_p0p1_reservation",
    "routersense_p0p1p2_hint",
    "routersense_joint_priority_phase_sync",
}


@dataclass(frozen=True)
class RunConfigSection:
    kind: str
    name: str


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    local_path: str = ""
    precision: str = "bf16"
    device_index: int = 0
    max_new_tokens: int = 32
    default_prompt: str = ""
    trace_layer_path: str = "auto"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class TopologyLauncherConfig:
    kind: str = "python"
    nnodes: int = 1
    nproc_per_node: int = 1
    standalone: bool = False
    master_port: int = 29500


@dataclass(frozen=True)
class TopologyConfig:
    launcher: TopologyLauncherConfig
    ep_size: int = 1
    network_scope: str = "single_node"
    interface_hint: str = ""


@dataclass(frozen=True)
class TokenizationConfig:
    padding: str = "longest"
    truncation: bool = False
    max_length: int | None = None
    expected_prompt_count: int | None = None
    expected_batch_rows: int | None = None
    expected_seq_len: int | None = None


@dataclass(frozen=True)
class WorkloadConfig:
    prompts: str = ""
    trace_artifact_dir: str = ""
    num_prompts: int | None = None
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    line: str = ""
    precision: str = "bf16"
    invariant_mode: str = "diagnostic"
    dispatcher: str = "alltoall"
    control_mode: str = "none"
    expert_compute_delay: float = 0.0
    scheduling_mode: str = "runtime_lookahead"


@dataclass(frozen=True)
class OnlinePolicyParameters:
    p0_weight: float = 1.0
    p1_reservation_weight: float = 1.0
    p2_hint_weight: float = 0.0
    residual_weight: float = 0.75
    barrier_weight: float = 1.75
    age_weight: float = 0.15
    prediction_weight: float = 0.35
    online_p2_predictor: str = "copy_current_dispatch"


@dataclass(frozen=True)
class OnlinePolicyP2Config:
    mode: str = "none"
    artifact: str = ""


@dataclass(frozen=True)
class OnlinePolicyConfig:
    name: str = "disabled"
    parameters: OnlinePolicyParameters = field(default_factory=OnlinePolicyParameters)
    p2: OnlinePolicyP2Config = field(default_factory=OnlinePolicyP2Config)


@dataclass(frozen=True)
class OfflineStudyWindowConfig:
    sample_selector: str = "first"
    start_layer_selector: str = "first"


@dataclass(frozen=True)
class OfflineStudyConfig:
    policies: tuple[str, ...] = ()
    reference_policies: tuple[str, ...] = ()
    p2_source: str = "zero_hint"
    window: OfflineStudyWindowConfig = field(default_factory=OfflineStudyWindowConfig)


@dataclass(frozen=True)
class ExecutionScheduleConfig:
    layer_selector: str = "all"
    phase_selector: str = "both"
    selected_layer_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "native_passthrough"
    bucket_mode: str = "dynamic_current"
    bucket_rows: int = 0
    safe_projection_mode: str = "host_select"
    preflight_mode: str = "full"
    schedule: ExecutionScheduleConfig = field(default_factory=ExecutionScheduleConfig)


@dataclass(frozen=True)
class ValidationConfig:
    save_logits: bool = False
    stop_after_selected_layer: bool = False
    allow_debug_capture: bool = False


@dataclass(frozen=True)
class ArtifactConfig:
    output_root: str


@dataclass(frozen=True)
class RunConfig:
    run: RunConfigSection
    model: ModelConfig
    topology: TopologyConfig
    workload: WorkloadConfig
    runtime: RuntimeConfig
    online_policy: OnlinePolicyConfig
    offline_study: OfflineStudyConfig
    execution: ExecutionConfig
    observation: RuntimeObservationConfig
    validation: ValidationConfig
    artifact: ArtifactConfig
    source_config_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _strict_str(value: Any, *, field_name: str, default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} must be a string")
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    return value


def _strict_bool(value: Any, *, field_name: str, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} must be a boolean")
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean, got {type(value).__name__}")


def _strict_int(value: Any, *, field_name: str, default: int | None = None, minimum: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} must be an integer")
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {type(value).__name__}")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return parsed


def _strict_optional_int(value: Any, *, field_name: str, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return _strict_int(value, field_name=field_name, minimum=minimum)


def _strict_float(value: Any, *, field_name: str, default: float | None = None, minimum: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} must be a finite number")
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number, got {type(value).__name__}")
    parsed = float(value)
    if not parsed == parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return parsed


def load_run_config(
    *,
    config_path: str | Path,
    overrides: list[str] | None = None,
    run_id: str | None = None,
    output_dir: str | None = None,
) -> RunConfig:
    root = _resolve_repo_root()
    config_path = (root / Path(config_path)).resolve() if not Path(config_path).is_absolute() else Path(config_path)
    payload = _load_yaml(config_path)
    payload = _resolve_nested_configs(payload, root)
    _validate_known_keys(payload)
    for override in overrides or []:
        _apply_override(payload, override)
    if run_id:
        payload.setdefault("run", {})["name"] = str(run_id)
    if output_dir:
        payload.setdefault("artifact", {})["output_root"] = str(output_dir)
    config = _build_run_config(payload, source_config_path=str(config_path))
    validate_run_config(config)
    return config


def validate_run_config(config: RunConfig) -> None:
    if config.run.kind not in SUPPORTED_RUN_KINDS:
        raise ValueError(f"unsupported run.kind {config.run.kind!r}")
    if config.observation.profile not in SUPPORTED_OBSERVATION_PROFILES:
        raise ValueError(f"unsupported observation.profile {config.observation.profile!r}")
    if config.runtime.control_mode not in SUPPORTED_CONTROL_MODES:
        raise ValueError(f"unsupported runtime.control_mode {config.runtime.control_mode!r}")
    if config.execution.mode not in SUPPORTED_EXECUTION_MODES:
        raise ValueError(f"unsupported execution.mode {config.execution.mode!r}")
    if (
        config.run.kind.startswith("online_")
        and config.online_policy.p2.mode == "none"
        and abs(float(config.online_policy.parameters.p2_hint_weight)) > 1e-9
    ):
        raise ValueError("p2_hint_weight must be 0 when p2_hint_mode=none")
    if config.online_policy.p2.mode == "deterministic_stub" and config.run.kind == "online_policy_correctness":
        pass
    if config.validation.stop_after_selected_layer and config.validation.save_logits:
        raise ValueError("save_logits must be false when stop_after_selected_layer=true")
    if config.observation.capture_enabled and config.observation.profile not in {"debug"} and not config.validation.allow_debug_capture:
        raise ValueError("selected tensor capture requires observation.profile=debug or explicit allow_debug_capture")
    if config.observation.capture_enabled and (not config.observation.capture_layer_selector or not config.observation.capture_phase_selector):
        raise ValueError("capture_enabled requires explicit capture_layer_selector and capture_phase_selector")
    if config.run.kind == "offline_trace":
        if config.online_policy.name != "disabled":
            raise ValueError("offline_trace does not accept online policy execution")
        if config.offline_study.policies:
            raise ValueError("offline_trace must not declare offline_study.policies")
        if config.topology.launcher.kind != "python":
            raise ValueError("offline_trace must not use torchrun topology")
    if config.run.kind == "offline_flow_study":
        if config.topology.launcher.kind != "python":
            raise ValueError("offline_flow_study must not use torchrun topology")
        if config.execution.mode != "native_passthrough":
            raise ValueError("offline_flow_study cannot use online execution mode")
        if config.online_policy.name != "disabled":
            raise ValueError("offline_flow_study must not declare online_policy.name")
        if not config.offline_study.policies:
            raise ValueError("offline_flow_study requires offline_study.policies")
    if config.run.kind == "online_observe":
        if config.online_policy.name != "disabled":
            raise ValueError("online_observe requires online_policy.name=disabled")
        if config.execution.mode != "native_passthrough":
            raise ValueError("online_observe requires native_passthrough execution")
        if config.offline_study.policies:
            raise ValueError("online_observe must not declare offline_study.policies")
    if config.run.kind == "online_policy_correctness":
        if config.online_policy.name in {"", "disabled"}:
            raise ValueError("online_policy_correctness requires a supported online_policy.name")
        if config.online_policy.name not in SUPPORTED_ONLINE_PHASE_POLICIES:
            raise ValueError(
                "online_policy_correctness only supports phase-local executable policies; "
                f"got {config.online_policy.name!r}"
            )
        if config.execution.mode not in {"phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"}:
            raise ValueError("online_policy_correctness requires phase_sync_wave, multiphase_pending_window, or joint_window_async_p2p execution")
        if config.runtime.control_mode != "sync_before_phase":
            raise ValueError("online_policy_correctness requires sync_before_phase control mode")
        if config.offline_study.policies:
            raise ValueError("online_policy_correctness must not declare offline_study.policies")
        if config.execution.mode == "multiphase_pending_window" and config.online_policy.name != "routersense_p0p1p2_hint":
            raise ValueError("multiphase_pending_window currently supports online_policy.name=routersense_p0p1p2_hint only")
        if config.execution.mode == "joint_window_async_p2p" and config.online_policy.name not in {
            "bucketed_fifo",
            "greedy_ready_set",
            "birkhoff_phase_local",
            "barrier_criticality_core_independent",
            "routersense_p0p1p2_hint",
        }:
            raise ValueError(
                "joint_window_async_p2p supports online_policy.name in "
                "{bucketed_fifo, greedy_ready_set, birkhoff_phase_local, barrier_criticality_core_independent, routersense_p0p1p2_hint}"
            )
    if config.execution.bucket_mode not in {"dynamic_current", "fixed_rows"}:
        raise ValueError(f"unsupported execution.bucket_mode {config.execution.bucket_mode!r}")
    if config.execution.safe_projection_mode not in {"disabled", "host_select"}:
        raise ValueError(f"unsupported execution.safe_projection_mode {config.execution.safe_projection_mode!r}")
    if config.execution.bucket_mode == "dynamic_current" and int(config.execution.bucket_rows) != 0:
        raise ValueError("dynamic_current requires execution.bucket_rows=0")
    if config.execution.bucket_mode == "fixed_rows":
        bucket_rows = int(config.execution.bucket_rows)
        if bucket_rows <= 0:
            raise ValueError("fixed_rows requires execution.bucket_rows > 0")
        if (bucket_rows & (bucket_rows - 1)) != 0:
            raise ValueError("fixed_rows requires execution.bucket_rows to be a power of two")
    if config.online_policy.name == "fast_bvn_single_tier" and int(config.topology.ep_size) > 8:
        raise ValueError("fast_bvn_single_tier supports EP size <= 8 only")
    _assert_no_credential_fields(config.to_dict())


def resolve_entrypoint_module(run_kind: str) -> str:
    mapping = {
        "offline_trace": "experiments.offline.collect_router_trace",
        "offline_flow_study": "experiments.offline.run_flow_schedule_study",
        "online_observe": "experiments.online.collect_native_ep_trace",
        "online_policy_correctness": "experiments.online.run_policy_correctness",
    }
    return mapping[run_kind]


def build_launch_command(
    *,
    config: RunConfig,
    config_path: str,
    overrides: list[str] | None = None,
    run_id: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    module = resolve_entrypoint_module(config.run.kind)
    extra = []
    if run_id:
        extra.extend(["--run-id", str(run_id)])
    if output_dir:
        extra.extend(["--output-dir", str(output_dir)])
    for override in overrides or []:
        extra.extend(["--override", override])
    if config.run.kind.startswith("online_"):
        launcher = config.topology.launcher
        command = [
            "torchrun",
            f"--nnodes={launcher.nnodes}",
            f"--nproc_per_node={launcher.nproc_per_node}",
            f"--master_port={launcher.master_port}",
        ]
        if launcher.standalone:
            command.append("--standalone")
        command.extend(["-m", module, "--config", str(config_path), *extra])
        return command
    return ["python", "-m", module, "--config", str(config_path), *extra]


def parse_override_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config at {path} must decode to a mapping")
    return payload


def _resolve_nested_configs(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved = dict(payload)
    for section in ("model", "topology"):
        section_payload = resolved.get(section, {})
        if isinstance(section_payload, dict) and "config" in section_payload:
            nested_path = root / str(section_payload["config"])
            nested = _load_yaml(nested_path)
            merged = dict(nested)
            for key, value in section_payload.items():
                if key != "config":
                    merged[key] = value
            resolved[section] = merged
    return resolved


def _validate_known_keys(payload: dict[str, Any]) -> None:
    allowed: dict[str, Any] = {
        "run": {"kind", "name"},
        "model": {"config", "model_id", "local_path", "precision", "device_index", "max_new_tokens", "default_prompt", "trace_layer_path", "trust_remote_code"},
        "topology": {
            "config",
            "launcher",
            "ep_size",
            "ep",
            "network_scope",
            "network",
            "interface_hint",
        },
        "workload": {"prompts", "trace_artifact_dir", "num_prompts", "tokenization"},
        "runtime": {"line", "precision", "invariant_mode", "dispatcher", "control_mode", "expert_compute_delay", "scheduling_mode"},
        "online_policy": {"name", "parameters", "p2"},
        "offline_study": {"policies", "reference_policies", "p2_source", "window"},
        "execution": {"mode", "bucket_mode", "bucket_rows", "safe_projection_mode", "preflight_mode", "schedule"},
        "evaluation": {"selected_layer_ids"},
        "observation": {
            "profile",
            "capture_enabled",
            "capture_expert_trace",
            "capture_layer_selector",
            "capture_phase_selector",
            "heartbeat_enabled",
            "per_wave_timing_enabled",
            "replay_trace_enabled",
        },
        "validation": {"save_logits", "stop_after_selected_layer", "allow_debug_capture"},
        "artifact": {"output_root", "artifact_root"},
        "requested_layer_selector": set(),
        "resolved_layer_selector": set(),
        "resolved_layer_ids": set(),
        "requested_preflight_mode": set(),
        "effective_preflight_mode": set(),
    }
    nested: dict[tuple[str, ...], set[str]] = {
        ("topology", "launcher"): {"kind", "nnodes", "nproc_per_node", "standalone", "master_port"},
        ("topology", "ep"): {"size"},
        ("topology", "network"): {"scope", "interface_hint"},
        ("online_policy", "parameters"): {
            "p0_weight",
            "p1_reservation_weight",
            "p2_hint_weight",
            "residual_weight",
            "barrier_weight",
            "age_weight",
            "prediction_weight",
            "online_p2_predictor",
        },
        ("online_policy", "p2"): {"mode", "artifact"},
        ("offline_study", "window"): {"sample_selector", "start_layer_selector"},
        ("execution", "schedule"): {"layer_selector", "phase_selector", "selected_layer_ids"},
        ("workload", "tokenization"): {
            "padding",
            "truncation",
            "max_length",
            "expected_prompt_count",
            "expected_batch_rows",
            "expected_seq_len",
        },
    }

    def _walk(mapping: dict[str, Any], path: tuple[str, ...] = ()) -> None:
        allowed_here = allowed.get(path[0], None) if len(path) == 1 else nested.get(path)
        if allowed_here is not None:
            unknown = set(mapping) - set(allowed_here)
            if unknown:
                location = ".".join(path) if path else "<root>"
                raise ValueError(f"unknown config keys under {location}: {sorted(unknown)!r}")
        elif not path:
            unknown = set(mapping) - set(allowed)
            if unknown:
                raise ValueError(f"unknown top-level config keys: {sorted(unknown)!r}")
        for key, value in mapping.items():
            if isinstance(value, dict):
                _walk(value, (*path, key))

    _walk(payload)


def _apply_override(payload: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must be key=value, got {override!r}")
    dotted_key, raw_value = override.split("=", 1)
    value = parse_override_value(raw_value)
    target = payload
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in target:
            raise ValueError(f"override path {dotted_key!r} does not exist")
        target = target[key]
        if not isinstance(target, dict):
            raise ValueError(f"override path {dotted_key!r} collides with non-mapping field")
    if keys[-1] not in target:
        raise ValueError(f"override path {dotted_key!r} does not exist")
    target[keys[-1]] = value


def _build_run_config(payload: dict[str, Any], *, source_config_path: str) -> RunConfig:
    run = _strict_mapping(payload.get("run"), field_name="run")
    model = _strict_mapping(payload.get("model"), field_name="model")
    topology = _strict_mapping(payload.get("topology"), field_name="topology")
    workload = _strict_mapping(payload.get("workload"), field_name="workload")
    tokenization = _strict_mapping(workload.get("tokenization"), field_name="workload.tokenization")
    runtime = _strict_mapping(payload.get("runtime"), field_name="runtime")
    online_policy = _strict_mapping(payload.get("online_policy"), field_name="online_policy")
    online_policy_parameters = _strict_mapping(online_policy.get("parameters"), field_name="online_policy.parameters")
    online_policy_p2 = _strict_mapping(online_policy.get("p2"), field_name="online_policy.p2")
    offline_study = _strict_mapping(payload.get("offline_study"), field_name="offline_study")
    offline_window = _strict_mapping(offline_study.get("window"), field_name="offline_study.window")
    execution = _strict_mapping(payload.get("execution"), field_name="execution")
    execution_schedule = _strict_mapping(execution.get("schedule"), field_name="execution.schedule")
    observation = _strict_mapping(payload.get("observation"), field_name="observation")
    validation = _strict_mapping(payload.get("validation"), field_name="validation")
    artifact = _strict_mapping(payload.get("artifact"), field_name="artifact")
    topology_launcher = _strict_mapping(topology.get("launcher"), field_name="topology.launcher")
    topology_ep = _strict_mapping(topology.get("ep"), field_name="topology.ep")
    topology_network = _strict_mapping(topology.get("network"), field_name="topology.network")
    evaluation = _strict_mapping(payload.get("evaluation"), field_name="evaluation")
    preflight_mode = _strict_str(execution.get("preflight_mode"), field_name="execution.preflight_mode", default="full")
    if preflight_mode not in {"full", "compact"}:
        raise ValueError(f"unsupported execution.preflight_mode: {preflight_mode!r}")
    return RunConfig(
        run=RunConfigSection(
            kind=_strict_str(run.get("kind"), field_name="run.kind", default=""),
            name=_strict_str(run.get("name"), field_name="run.name", default=""),
        ),
        model=ModelConfig(
            model_id=_strict_str(model.get("model_id"), field_name="model.model_id", default=""),
            local_path=_strict_str(model.get("local_path"), field_name="model.local_path", default=""),
            precision=_strict_str(model.get("precision", runtime.get("precision", "bf16")), field_name="model.precision"),
            device_index=_strict_int(model.get("device_index"), field_name="model.device_index", default=0, minimum=0),
            max_new_tokens=_strict_int(model.get("max_new_tokens"), field_name="model.max_new_tokens", default=32, minimum=0),
            default_prompt=_strict_str(model.get("default_prompt"), field_name="model.default_prompt", default=""),
            trace_layer_path=_strict_str(model.get("trace_layer_path"), field_name="model.trace_layer_path", default="auto"),
            trust_remote_code=_strict_bool(model.get("trust_remote_code"), field_name="model.trust_remote_code", default=False),
        ),
        topology=TopologyConfig(
            launcher=TopologyLauncherConfig(
                kind=_strict_str(topology_launcher.get("kind"), field_name="topology.launcher.kind", default="python"),
                nnodes=_strict_int(topology_launcher.get("nnodes"), field_name="topology.launcher.nnodes", default=1, minimum=1),
                nproc_per_node=_strict_int(
                    topology_launcher.get("nproc_per_node"),
                    field_name="topology.launcher.nproc_per_node",
                    default=1,
                    minimum=1,
                ),
                standalone=_strict_bool(topology_launcher.get("standalone"), field_name="topology.launcher.standalone", default=False),
                master_port=_strict_int(topology_launcher.get("master_port"), field_name="topology.launcher.master_port", default=29500, minimum=1),
            ),
            ep_size=_strict_int(topology.get("ep_size", topology_ep.get("size", 1)), field_name="topology.ep_size", minimum=1),
            network_scope=_strict_str(
                topology.get("network_scope", topology_network.get("scope", "single_node")),
                field_name="topology.network_scope",
            ),
            interface_hint=_strict_str(
                topology.get("interface_hint", topology_network.get("interface_hint", "")),
                field_name="topology.interface_hint",
            ),
        ),
        workload=WorkloadConfig(
            prompts=_strict_str(workload.get("prompts"), field_name="workload.prompts", default=""),
            trace_artifact_dir=_strict_str(workload.get("trace_artifact_dir"), field_name="workload.trace_artifact_dir", default=""),
            num_prompts=_strict_optional_int(workload.get("num_prompts"), field_name="workload.num_prompts", minimum=0),
            tokenization=TokenizationConfig(
                padding=_strict_str(tokenization.get("padding"), field_name="workload.tokenization.padding", default="longest"),
                truncation=_strict_bool(tokenization.get("truncation"), field_name="workload.tokenization.truncation", default=False),
                max_length=_strict_optional_int(tokenization.get("max_length"), field_name="workload.tokenization.max_length", minimum=0),
                expected_prompt_count=_strict_optional_int(
                    tokenization.get("expected_prompt_count"),
                    field_name="workload.tokenization.expected_prompt_count",
                    minimum=0,
                ),
                expected_batch_rows=_strict_optional_int(
                    tokenization.get("expected_batch_rows"),
                    field_name="workload.tokenization.expected_batch_rows",
                    minimum=0,
                ),
                expected_seq_len=_strict_optional_int(
                    tokenization.get("expected_seq_len"),
                    field_name="workload.tokenization.expected_seq_len",
                    minimum=0,
                ),
            ),
        ),
        runtime=RuntimeConfig(
            line=_strict_str(runtime.get("line"), field_name="runtime.line", default=""),
            precision=_strict_str(runtime.get("precision", model.get("precision", "bf16")), field_name="runtime.precision"),
            invariant_mode=_strict_str(runtime.get("invariant_mode"), field_name="runtime.invariant_mode", default="diagnostic"),
            dispatcher=_strict_str(runtime.get("dispatcher"), field_name="runtime.dispatcher", default="alltoall"),
            control_mode=_strict_str(runtime.get("control_mode"), field_name="runtime.control_mode", default="none"),
            expert_compute_delay=_strict_float(runtime.get("expert_compute_delay"), field_name="runtime.expert_compute_delay", default=0.0, minimum=0.0),
            scheduling_mode=_strict_str(runtime.get("scheduling_mode"), field_name="runtime.scheduling_mode", default="runtime_lookahead"),
        ),
        online_policy=OnlinePolicyConfig(
            name=_strict_str(online_policy.get("name"), field_name="online_policy.name", default="disabled"),
            parameters=OnlinePolicyParameters(
                p0_weight=_strict_float(online_policy_parameters.get("p0_weight"), field_name="online_policy.parameters.p0_weight", default=1.0),
                p1_reservation_weight=_strict_float(
                    online_policy_parameters.get("p1_reservation_weight"),
                    field_name="online_policy.parameters.p1_reservation_weight",
                    default=1.0,
                ),
                p2_hint_weight=_strict_float(online_policy_parameters.get("p2_hint_weight"), field_name="online_policy.parameters.p2_hint_weight", default=0.0),
                residual_weight=_strict_float(online_policy_parameters.get("residual_weight"), field_name="online_policy.parameters.residual_weight", default=0.75),
                barrier_weight=_strict_float(online_policy_parameters.get("barrier_weight"), field_name="online_policy.parameters.barrier_weight", default=1.75),
                age_weight=_strict_float(online_policy_parameters.get("age_weight"), field_name="online_policy.parameters.age_weight", default=0.15),
                prediction_weight=_strict_float(
                    online_policy_parameters.get("prediction_weight"),
                    field_name="online_policy.parameters.prediction_weight",
                    default=0.35,
                ),
                online_p2_predictor=_strict_str(
                    online_policy_parameters.get("online_p2_predictor"),
                    field_name="online_policy.parameters.online_p2_predictor",
                    default="copy_current_dispatch",
                ),
            ),
            p2=OnlinePolicyP2Config(
                mode=_strict_str(online_policy_p2.get("mode"), field_name="online_policy.p2.mode", default="none"),
                artifact=_strict_str(online_policy_p2.get("artifact"), field_name="online_policy.p2.artifact", default=""),
            ),
        ),
        offline_study=OfflineStudyConfig(
            policies=tuple(str(item) for item in offline_study.get("policies", []) or ()),
            reference_policies=tuple(str(item) for item in offline_study.get("reference_policies", []) or ()),
            p2_source=_strict_str(offline_study.get("p2_source"), field_name="offline_study.p2_source", default="zero_hint"),
            window=OfflineStudyWindowConfig(
                sample_selector=_strict_str(offline_window.get("sample_selector"), field_name="offline_study.window.sample_selector", default="first"),
                start_layer_selector=_strict_str(
                    offline_window.get("start_layer_selector"),
                    field_name="offline_study.window.start_layer_selector",
                    default="first",
                ),
            ),
        ),
        execution=ExecutionConfig(
            mode=_strict_str(execution.get("mode"), field_name="execution.mode", default="native_passthrough"),
            bucket_mode=_strict_str(execution.get("bucket_mode"), field_name="execution.bucket_mode", default="dynamic_current"),
            bucket_rows=_strict_int(execution.get("bucket_rows"), field_name="execution.bucket_rows", default=0, minimum=0),
            safe_projection_mode=_strict_str(execution.get("safe_projection_mode"), field_name="execution.safe_projection_mode", default="host_select"),
            preflight_mode=preflight_mode,
            schedule=ExecutionScheduleConfig(
                layer_selector=_strict_str(execution_schedule.get("layer_selector"), field_name="execution.schedule.layer_selector", default="all"),
                phase_selector=_strict_str(execution_schedule.get("phase_selector"), field_name="execution.schedule.phase_selector", default="both"),
                selected_layer_ids=tuple(
                    str(item)
                    for item in (
                        execution_schedule.get("selected_layer_ids")
                        or evaluation.get("selected_layer_ids")
                        or ()
                    )
                ),
            ),
        ),
        observation=RuntimeObservationConfig(
            profile=_strict_str(observation.get("profile"), field_name="observation.profile", default="minimal"),
            invariant_mode=_strict_str(observation.get("invariant_mode", runtime.get("invariant_mode", "diagnostic")), field_name="observation.invariant_mode"),
            capture_enabled=_strict_bool(observation.get("capture_enabled"), field_name="observation.capture_enabled", default=False),
            capture_expert_trace=_strict_bool(
                observation.get("capture_expert_trace"),
                field_name="observation.capture_expert_trace",
                default=False,
            ),
            capture_layer_selector=_strict_str(observation.get("capture_layer_selector"), field_name="observation.capture_layer_selector", default=""),
            capture_phase_selector=_strict_str(observation.get("capture_phase_selector"), field_name="observation.capture_phase_selector", default=""),
            heartbeat_enabled=_strict_bool(observation.get("heartbeat_enabled"), field_name="observation.heartbeat_enabled", default=False),
            per_wave_timing_enabled=_strict_bool(
                observation.get("per_wave_timing_enabled"),
                field_name="observation.per_wave_timing_enabled",
                default=False,
            ),
            replay_trace_enabled=_strict_bool(observation.get("replay_trace_enabled"), field_name="observation.replay_trace_enabled", default=False),
        ),
        validation=ValidationConfig(
            save_logits=_strict_bool(validation.get("save_logits"), field_name="validation.save_logits", default=False),
            stop_after_selected_layer=_strict_bool(
                validation.get("stop_after_selected_layer"),
                field_name="validation.stop_after_selected_layer",
                default=False,
            ),
            allow_debug_capture=_strict_bool(
                validation.get("allow_debug_capture"),
                field_name="validation.allow_debug_capture",
                default=False,
            ),
        ),
        artifact=ArtifactConfig(
            output_root=_strict_str(
                artifact.get("output_root", artifact.get("artifact_root", "")),
                field_name="artifact.output_root",
                default="",
            )
        ),
        source_config_path=source_config_path,
    )


def _assert_no_credential_fields(payload: Any, *, path: str = "") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            full = f"{path}.{key}" if path else str(key)
            if str(key).lower() in {"password", "passwd", "token", "secret", "credential"}:
                raise ValueError(f"credential-like field is not allowed in formal config: {full}")
            _assert_no_credential_fields(value, path=full)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_no_credential_fields(value, path=f"{path}[{index}]")
