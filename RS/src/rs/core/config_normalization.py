from __future__ import annotations

from dataclasses import asdict, dataclass, field
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


def normalize_run_config(raw_config: dict[str, Any]) -> CanonicalRunConfig:
    payload = dict(raw_config)
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
            "bucket_rows": int(config.traffic.get("bucket_rows", 0)),
            "p0_weight": float((config.policy.get("options", {}) or {}).get("p0_weight", 1.0)),
            "p1_reservation_weight": float((config.policy.get("options", {}) or {}).get("p1_reservation_weight", 1.0)),
            "p2_hint_weight": float((config.policy.get("options", {}) or {}).get("p2_hint_weight", 1.0)),
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
]
