from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConfigMigration:
    source_field: str
    target_field: str
    transform: str
    deprecated_since: str
    removal_after: str | None


@dataclass(frozen=True)
class CanonicalRunConfig:
    schema_version: int
    run: dict[str, Any]
    model: dict[str, Any]
    topology: dict[str, Any]
    workload: dict[str, Any]
    runtime: dict[str, Any]
    traffic: dict[str, Any]
    policy: dict[str, Any]
    prediction: dict[str, Any]
    evaluation: dict[str, Any]
    replay: dict[str, Any]
    oracle: dict[str, Any]
    regime_analysis: dict[str, Any]
    strategies: tuple[dict[str, Any], ...] = ()
    raw_kind: str = ""
    applied_migrations: tuple[ConfigMigration, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["applied_migrations"] = [asdict(item) for item in self.applied_migrations]
        return payload


_COMPONENT_ALLOWED_OVERRIDE_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    ("model",): frozenset({"component", "path", "local_path"}),
    ("topology",): frozenset({"component"}),
    ("workload",): frozenset({"component"}),
    ("runtime",): frozenset({"component"}),
}


def _ensure_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping for {field_name}, got {type(value).__name__}")
    return dict(value)


def _canonical_list(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _conflict(name: str, old_value: Any, new_value: Any) -> None:
    if old_value != new_value:
        raise ValueError(f"conflicting config values for {name}: legacy={old_value!r} canonical={new_value!r}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(dict(current), dict(value))
        else:
            merged[key] = value
    return merged


def resolve_config_components(
    raw_config: dict[str, Any],
    *,
    source_path: str | Path | None,
) -> dict[str, Any]:
    if source_path is None:
        return dict(raw_config)
    config_path = Path(source_path).resolve()
    visited: set[Path] = set()

    def _load_yaml_mapping(path: Path) -> dict[str, Any]:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected mapping config component: {path}")
        return dict(payload)

    def _resolve_component_path(component_ref: str, *, base_dir: Path) -> Path:
        candidate = Path(component_ref)
        search_paths: list[Path] = []
        if candidate.is_absolute():
            search_paths.append(candidate)
        else:
            current = base_dir
            while True:
                search_paths.append((current / candidate).resolve())
                if current.parent == current:
                    break
                current = current.parent
            search_paths.append(candidate.resolve())
        for resolved in search_paths:
            if resolved.exists():
                return resolved
        raise ValueError(f"missing config component: {component_ref!r} resolved from {base_dir}")

    def _resolve_mapping(mapping: dict[str, Any], *, current_path: Path, path: tuple[str, ...]) -> dict[str, Any]:
        resolved = dict(mapping)
        component_key = None
        component_ref: str | None = None
        if isinstance(resolved.get("component"), str) and str(resolved["component"]).strip():
            component_key = "component"
            component_ref = str(resolved["component"]).strip()
        elif path == ("model",):
            for candidate_key in ("path", "local_path"):
                candidate_value = resolved.get(candidate_key)
                if not isinstance(candidate_value, str):
                    continue
                candidate_ref = str(candidate_value).strip()
                if candidate_ref.endswith((".yaml", ".yml")):
                    component_key = candidate_key
                    component_ref = candidate_ref
                    break

        if component_key is not None and component_ref is not None:
            allowed = _COMPONENT_ALLOWED_OVERRIDE_FIELDS.get(path, frozenset({"component"}))
            invalid_keys = sorted(key for key in resolved if key in {"component", "path"} and key not in allowed)
            if invalid_keys:
                joined = ".".join(path) or "<root>"
                raise ValueError(f"unsupported component indirection field(s) at {joined}: {invalid_keys}")
            component_path = _resolve_component_path(component_ref, base_dir=current_path.parent)
            if component_path in visited:
                cycle = " -> ".join(str(item) for item in (*visited, component_path))
                raise ValueError(f"cyclic config component reference detected: {cycle}")
            visited.add(component_path)
            component_payload = _resolve_mapping(_load_yaml_mapping(component_path), current_path=component_path, path=path)
            visited.remove(component_path)
            overrides = {key: value for key, value in resolved.items() if key != component_key}
            if component_key == "path":
                overrides.pop("component", None)
            if component_key == "local_path":
                resolved = _deep_merge(overrides, component_payload)
            else:
                resolved = _deep_merge(component_payload, overrides)

        for key, value in tuple(resolved.items()):
            if isinstance(value, dict):
                resolved[key] = _resolve_mapping(value, current_path=current_path, path=(*path, str(key)))
        return resolved

    return _resolve_mapping(dict(raw_config), current_path=config_path, path=())


def normalize_run_config(raw_config: dict[str, Any], *, source_path: str | Path | None = None) -> CanonicalRunConfig:
    payload = resolve_config_components(dict(raw_config), source_path=source_path)
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version >= 1:
        return _normalize_v1(payload, schema_version=schema_version)
    return _normalize_v0(payload)


def _normalize_v1(payload: dict[str, Any], *, schema_version: int) -> CanonicalRunConfig:
    legacy_only_keys = {
        "fixture_dir",
        "max_windows",
        "bucket_rows",
        "policies",
        "hints",
        "scheduling_mode",
        "expert_compute_delay",
        "output_dir",
        "execution",
        "comparison",
    }
    mixed = sorted(key for key in legacy_only_keys if key in payload)
    if mixed:
        raise ValueError(f"schema_version=1 config must not mix legacy fields: {mixed}")
    run = _ensure_mapping(payload.get("run"), field_name="run")
    model = _ensure_mapping(payload.get("model"), field_name="model")
    topology = _ensure_mapping(payload.get("topology"), field_name="topology")
    workload = _ensure_mapping(payload.get("workload"), field_name="workload")
    runtime = _ensure_mapping(payload.get("runtime"), field_name="runtime")
    traffic = _ensure_mapping(payload.get("traffic"), field_name="traffic")
    policy = _ensure_mapping(payload.get("policy"), field_name="policy")
    prediction = _ensure_mapping(payload.get("prediction"), field_name="prediction")
    evaluation = _ensure_mapping(payload.get("evaluation"), field_name="evaluation")
    replay = _ensure_mapping(payload.get("replay"), field_name="replay")
    oracle = _ensure_mapping(payload.get("oracle"), field_name="oracle")
    regime_analysis = _ensure_mapping(payload.get("regime_analysis"), field_name="regime_analysis")
    strategies = tuple(dict(item) for item in _canonical_list(payload.get("strategies")) if isinstance(item, dict))
    return CanonicalRunConfig(
        schema_version=schema_version,
        run=run,
        model=model,
        topology=topology,
        workload=workload,
        runtime=runtime,
        traffic=traffic,
        policy=policy,
        prediction=prediction,
        evaluation=evaluation,
        replay=replay,
        oracle=oracle,
        regime_analysis=regime_analysis,
        strategies=strategies,
        raw_kind=str(run.get("kind", "")),
        applied_migrations=(),
    )


def _normalize_v0(payload: dict[str, Any]) -> CanonicalRunConfig:
    migrations: list[ConfigMigration] = []
    if "fixture_dir" in payload:
        # offline replay legacy shape
        model = _ensure_mapping(payload.get("model"), field_name="model")
        topology = _ensure_mapping(payload.get("topology"), field_name="topology")
        workload = _ensure_mapping(payload.get("workload"), field_name="workload")
        runtime = {
            "line": str(_ensure_mapping(payload.get("runtime"), field_name="runtime").get("line", "offline_replay")),
        }
        traffic = {
            "matrix_unit": "rows",
            "bucket_rows": [int(value) for value in _canonical_list(payload.get("bucket_rows", [1024]))],
        }
        policy = {"names": [str(item) for item in _canonical_list(payload.get("policies"))]}
        prediction = {"names": [str(item) for item in _canonical_list(payload.get("hints"))]}
        evaluation = {
            "max_windows": int(payload.get("max_windows", 4)),
            "metrics": ["makespan", "plan_digest"],
        }
        replay = {
            "fixture_dir": str(payload.get("fixture_dir", "")),
            "output_dir": str(payload.get("output_dir", "outputs/offline/offline_replay_smoke")),
            "scheduling_mode": str(payload.get("scheduling_mode", "execution_window")),
            "expert_compute_delay": float(payload.get("expert_compute_delay", 0.0)),
        }
        migrations.extend(
            (
                ConfigMigration("fixture_dir", "replay.fixture_dir", "identity", "v1", None),
                ConfigMigration("bucket_rows", "traffic.bucket_rows", "identity", "v1", None),
                ConfigMigration("policies", "policy.names", "identity", "v1", None),
                ConfigMigration("hints", "prediction.names", "identity", "v1", None),
            )
        )
        return CanonicalRunConfig(
            schema_version=1,
            run={"kind": "offline_replay", "name": str(payload.get("run_name", "offline_replay"))},
            model=model,
            topology=topology,
            workload=workload,
            runtime=runtime,
            traffic=traffic,
            policy=policy,
            prediction=prediction,
            evaluation=evaluation,
            replay=replay,
            oracle={},
            regime_analysis={},
            strategies=(),
            raw_kind="offline_replay",
            applied_migrations=tuple(migrations),
        )
    # online comparison legacy shape
    model = _ensure_mapping(payload.get("model"), field_name="model")
    topology = _ensure_mapping(payload.get("topology"), field_name="topology")
    workload = _ensure_mapping(payload.get("workload"), field_name="workload")
    runtime_legacy = _ensure_mapping(payload.get("runtime"), field_name="runtime")
    execution = _ensure_mapping(payload.get("execution"), field_name="execution")
    comparison = _ensure_mapping(payload.get("comparison"), field_name="comparison")
    strategies = tuple(dict(item) if isinstance(item, dict) else {"name": str(item)} for item in _canonical_list(payload.get("strategies")))
    migrations.extend(
        (
            ConfigMigration("execution.bucket_rows", "traffic.bucket_rows", "identity", "v1", None),
            ConfigMigration("execution.repetitions", "evaluation.repeats", "identity", "v1", None),
            ConfigMigration("comparison.baseline_strategy", "evaluation.baseline_strategy", "identity", "v1", None),
        )
    )
    return CanonicalRunConfig(
        schema_version=1,
        run={"kind": "online_strategy_comparison", "name": str(payload.get("run_name", "online_strategy_comparison"))},
        model=model,
        topology=topology,
        workload=workload,
        runtime={
            "line": str(runtime_legacy.get("line", "phase_sync")),
            "output_mode": str(runtime_legacy.get("output_mode", "paper")),
            "precision": str(runtime_legacy.get("precision", "fp16")),
            "dispatcher": str(runtime_legacy.get("dispatcher", "alltoall")),
            "selected_layers": str(execution.get("schedule_layer_selector", "all")),
        },
        traffic={
            "matrix_unit": "rows",
            "bucket_rows": int(execution.get("bucket_rows", 0)),
        },
        policy={
            "options": {
                "p0_weight": float(execution.get("p0_weight", 1.0)),
                "p1_reservation_weight": float(execution.get("p1_reservation_weight", 1.0)),
                "p2_hint_weight": float(execution.get("p2_hint_weight", 1.0)),
            }
        },
        prediction={},
        evaluation={
            "repeats": int(execution.get("repetitions", 1)),
            "warmup": int(execution.get("warmup", 0)),
            "metrics": tuple(str(item) for item in _canonical_list(comparison.get("metrics"))),
            "baseline_strategy": str(comparison.get("baseline_strategy", "")),
            "phase_selector": str(execution.get("schedule_phase_selector", "both")),
        },
        replay={},
        oracle={},
        regime_analysis={},
        strategies=strategies,
        raw_kind="online_strategy_comparison",
        applied_migrations=tuple(migrations),
    )


def canonical_offline_replay_payload(config: CanonicalRunConfig) -> dict[str, Any]:
    if config.raw_kind != "offline_replay":
        raise ValueError(f"expected offline_replay config, got {config.raw_kind!r}")
    return {
        "schema_version": 1,
        "run": config.run,
        "model": config.model,
        "topology": config.topology,
        "workload": config.workload,
        "runtime": config.runtime,
        "traffic": config.traffic,
        "policy": config.policy,
        "prediction": config.prediction,
        "evaluation": config.evaluation,
        "replay": config.replay,
        "oracle": config.oracle,
        "regime_analysis": config.regime_analysis,
        "applied_migrations": [asdict(item) for item in config.applied_migrations],
    }


def legacy_offline_replay_payload(config: CanonicalRunConfig) -> dict[str, Any]:
    if config.raw_kind != "offline_replay":
        raise ValueError(f"expected offline_replay config, got {config.raw_kind!r}")
    return {
        "fixture_dir": str(config.replay.get("fixture_dir", "")),
        "max_windows": int(config.evaluation.get("max_windows", 4)),
        "bucket_rows": list(config.traffic.get("bucket_rows", [])),
        "policies": list(config.policy.get("names", [])),
        "hints": list(config.prediction.get("names", [])),
        "scheduling_mode": str(config.replay.get("scheduling_mode", "execution_window")),
        "expert_compute_delay": float(config.replay.get("expert_compute_delay", 0.0)),
        "output_dir": str(config.replay.get("output_dir", "outputs/offline/offline_replay_smoke")),
    }


def canonical_online_comparison_payload(config: CanonicalRunConfig) -> dict[str, Any]:
    if config.raw_kind != "online_strategy_comparison":
        raise ValueError(f"expected online_strategy_comparison config, got {config.raw_kind!r}")
    return {
        "schema_version": 1,
        "run": config.run,
        "model": config.model,
        "topology": config.topology,
        "workload": config.workload,
        "runtime": config.runtime,
        "traffic": config.traffic,
        "policy": config.policy,
        "prediction": config.prediction,
        "evaluation": config.evaluation,
        "strategies": [dict(item) for item in config.strategies],
        "applied_migrations": [asdict(item) for item in config.applied_migrations],
    }


def legacy_online_comparison_payload(config: CanonicalRunConfig) -> dict[str, Any]:
    if config.raw_kind != "online_strategy_comparison":
        raise ValueError(f"expected online_strategy_comparison config, got {config.raw_kind!r}")
    return {
        "model": config.model,
        "topology": config.topology,
        "runtime": {
            "line": str(config.runtime.get("line", "phase_sync")),
            "output_mode": str(config.runtime.get("output_mode", "paper")),
            "precision": str(config.runtime.get("precision", "fp16")),
            "dispatcher": str(config.runtime.get("dispatcher", "alltoall")),
        },
        "workload": config.workload,
        "strategies": [dict(item) for item in config.strategies],
        "execution": {
            "repetitions": int(config.evaluation.get("repeats", 1)),
            "warmup": int(config.evaluation.get("warmup", 0)),
            "bucket_mode": str(config.traffic.get("bucket_mode", "dynamic_current")),
            "bucket_rows": int(config.traffic.get("bucket_rows", 0)),
            "safe_projection_mode": str(config.policy.get("options", {}).get("safe_projection_mode", "host_select")),
            "p0_weight": float((config.policy.get("options", {}) or {}).get("p0_weight", 1.0)),
            "p1_reservation_weight": float((config.policy.get("options", {}) or {}).get("p1_reservation_weight", 1.0)),
            "p2_hint_weight": float((config.policy.get("options", {}) or {}).get("p2_hint_weight", 1.0)),
            "residual_weight": float((config.policy.get("options", {}) or {}).get("residual_weight", 0.75)),
            "barrier_weight": float((config.policy.get("options", {}) or {}).get("barrier_weight", 1.75)),
            "age_weight": float((config.policy.get("options", {}) or {}).get("age_weight", 0.15)),
            "prediction_weight": float((config.policy.get("options", {}) or {}).get("prediction_weight", 0.35)),
            "schedule_layer_selector": str(config.runtime.get("selected_layers", "all")),
            "schedule_phase_selector": str(config.evaluation.get("phase_selector", "both")),
        },
        "comparison": {
            "baseline_strategy": str(config.evaluation.get("baseline_strategy", "")),
            "metrics": list(config.evaluation.get("metrics", ())),
        },
    }


__all__ = [
    "CanonicalRunConfig",
    "ConfigMigration",
    "canonical_offline_replay_payload",
    "canonical_online_comparison_payload",
    "legacy_offline_replay_payload",
    "legacy_online_comparison_payload",
    "normalize_run_config",
    "resolve_config_components",
]
