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

SUPPORTED_OBSERVATION_PROFILES = {"minimal", "perf", "execution", "debug"}
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
    run = payload.get("run", {})
    model = payload.get("model", {})
    topology = payload.get("topology", {})
    workload = payload.get("workload", {})
    tokenization = workload.get("tokenization", {}) or {}
    runtime = payload.get("runtime", {})
    online_policy = payload.get("online_policy", {})
    offline_study = payload.get("offline_study", {})
    execution = payload.get("execution", {})
    observation = payload.get("observation", {})
    validation = payload.get("validation", {})
    artifact = payload.get("artifact", {})
    topology_launcher = topology.get("launcher", {})
    preflight_mode = str(execution.get("preflight_mode", "full"))
    if preflight_mode not in {"full", "compact"}:
        raise ValueError(f"unsupported execution.preflight_mode: {preflight_mode!r}")
    return RunConfig(
        run=RunConfigSection(kind=str(run.get("kind", "")), name=str(run.get("name", ""))),
        model=ModelConfig(
            model_id=str(model.get("model_id", "")),
            local_path=str(model.get("local_path", "")),
            precision=str(model.get("precision", runtime.get("precision", "bf16"))),
            device_index=int(model.get("device_index", 0)),
            max_new_tokens=int(model.get("max_new_tokens", 32)),
            default_prompt=str(model.get("default_prompt", "")),
            trace_layer_path=str(model.get("trace_layer_path", "auto")),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
        ),
        topology=TopologyConfig(
            launcher=TopologyLauncherConfig(
                kind=str(topology_launcher.get("kind", "python")),
                nnodes=int(topology_launcher.get("nnodes", 1)),
                nproc_per_node=int(topology_launcher.get("nproc_per_node", 1)),
                standalone=bool(topology_launcher.get("standalone", False)),
                master_port=int(topology_launcher.get("master_port", 29500)),
            ),
            ep_size=int(topology.get("ep_size", topology.get("ep", {}).get("size", 1))),
            network_scope=str(topology.get("network_scope", topology.get("network", {}).get("scope", "single_node"))),
            interface_hint=str(topology.get("interface_hint", topology.get("network", {}).get("interface_hint", ""))),
        ),
        workload=WorkloadConfig(
            prompts=str(workload.get("prompts", "")),
            trace_artifact_dir=str(workload.get("trace_artifact_dir", "")),
            num_prompts=None if workload.get("num_prompts") is None else int(workload.get("num_prompts")),
            tokenization=TokenizationConfig(
                padding=str(tokenization.get("padding", "longest")),
                truncation=bool(tokenization.get("truncation", False)),
                max_length=None if tokenization.get("max_length") is None else int(tokenization.get("max_length")),
                expected_prompt_count=(
                    None
                    if tokenization.get("expected_prompt_count") is None
                    else int(tokenization.get("expected_prompt_count"))
                ),
                expected_batch_rows=(
                    None
                    if tokenization.get("expected_batch_rows") is None
                    else int(tokenization.get("expected_batch_rows"))
                ),
                expected_seq_len=(
                    None if tokenization.get("expected_seq_len") is None else int(tokenization.get("expected_seq_len"))
                ),
            ),
        ),
        runtime=RuntimeConfig(
            line=str(runtime.get("line", "")),
            precision=str(runtime.get("precision", model.get("precision", "bf16"))),
            invariant_mode=str(runtime.get("invariant_mode", "diagnostic")),
            dispatcher=str(runtime.get("dispatcher", "alltoall")),
            control_mode=str(runtime.get("control_mode", "none")),
            expert_compute_delay=float(runtime.get("expert_compute_delay", 0.0)),
            scheduling_mode=str(runtime.get("scheduling_mode", "runtime_lookahead")),
        ),
        online_policy=OnlinePolicyConfig(
            name=str(online_policy.get("name", "disabled")),
            parameters=OnlinePolicyParameters(
                p0_weight=float(online_policy.get("parameters", {}).get("p0_weight", 1.0)),
                p1_reservation_weight=float(online_policy.get("parameters", {}).get("p1_reservation_weight", 1.0)),
                p2_hint_weight=float(online_policy.get("parameters", {}).get("p2_hint_weight", 0.0)),
                residual_weight=float(online_policy.get("parameters", {}).get("residual_weight", 0.75)),
                barrier_weight=float(online_policy.get("parameters", {}).get("barrier_weight", 1.75)),
                age_weight=float(online_policy.get("parameters", {}).get("age_weight", 0.15)),
                prediction_weight=float(online_policy.get("parameters", {}).get("prediction_weight", 0.35)),
                online_p2_predictor=str(online_policy.get("parameters", {}).get("online_p2_predictor", "copy_current_dispatch")),
            ),
            p2=OnlinePolicyP2Config(
                mode=str(online_policy.get("p2", {}).get("mode", "none")),
                artifact=str(online_policy.get("p2", {}).get("artifact", "")),
            ),
        ),
        offline_study=OfflineStudyConfig(
            policies=tuple(str(item) for item in offline_study.get("policies", []) or ()),
            reference_policies=tuple(str(item) for item in offline_study.get("reference_policies", []) or ()),
            p2_source=str(offline_study.get("p2_source", "zero_hint")),
            window=OfflineStudyWindowConfig(
                sample_selector=str(offline_study.get("window", {}).get("sample_selector", "first")),
                start_layer_selector=str(offline_study.get("window", {}).get("start_layer_selector", "first")),
            ),
        ),
        execution=ExecutionConfig(
            mode=str(execution.get("mode", "native_passthrough")),
            bucket_mode=str(execution.get("bucket_mode", "dynamic_current")),
            bucket_rows=int(execution.get("bucket_rows", 0)),
            safe_projection_mode=str(execution.get("safe_projection_mode", "host_select")),
            preflight_mode=preflight_mode,
            schedule=ExecutionScheduleConfig(
                layer_selector=str(execution.get("schedule", {}).get("layer_selector", "all")),
                phase_selector=str(execution.get("schedule", {}).get("phase_selector", "both")),
                selected_layer_ids=tuple(
                    str(item)
                    for item in (
                        execution.get("schedule", {}).get("selected_layer_ids")
                        or payload.get("evaluation", {}).get("selected_layer_ids")
                        or ()
                    )
                ),
            ),
        ),
        observation=RuntimeObservationConfig(
            profile=str(observation.get("profile", "minimal")),
            invariant_mode=str(observation.get("invariant_mode", runtime.get("invariant_mode", "diagnostic"))),
            capture_enabled=bool(observation.get("capture_enabled", False)),
            capture_expert_trace=bool(observation.get("capture_expert_trace", False)),
            capture_layer_selector=str(observation.get("capture_layer_selector", "")),
            capture_phase_selector=str(observation.get("capture_phase_selector", "")),
            heartbeat_enabled=bool(observation.get("heartbeat_enabled", False)),
            per_wave_timing_enabled=bool(observation.get("per_wave_timing_enabled", False)),
            replay_trace_enabled=bool(observation.get("replay_trace_enabled", False)),
        ),
        validation=ValidationConfig(
            save_logits=bool(validation.get("save_logits", False)),
            stop_after_selected_layer=bool(validation.get("stop_after_selected_layer", False)),
            allow_debug_capture=bool(validation.get("allow_debug_capture", False)),
        ),
        artifact=ArtifactConfig(output_root=str(artifact.get("output_root", artifact.get("artifact_root", "")))),
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
