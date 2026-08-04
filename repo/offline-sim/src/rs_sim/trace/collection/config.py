"""Configuration loading and validation for the independent trace pipeline."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from rs_sim.contracts.paper_defaults import (
    PAPER_MAX_TASK_BYTES, PAPER_P0_P1_COMPUTE_END_BARRIER, PAPER_RELEASE_MODE,
)
from typing import Any

SCHEMA_VERSION = "RS_SIM_TRACE_CAPTURE_PIPELINE"
SUPPORTED_BACKENDS = {"MEGATRON_CORE_AUTO", "EXPLICIT_API", "REPLAY_ONLY"}
SUPPORTED_SPLITS = {"train", "validation", "test"}


class CaptureConfigError(ValueError):
    pass


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaptureConfigError(f"{name} must be an object")
    return value


def _require_nonempty(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CaptureConfigError(f"{name} must be non-empty")
    return text




def _reject_unknown(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise CaptureConfigError(f"{name} contains unknown fields: {', '.join(unknown)}")


def _require_bool(value: Any, name: str, *, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return bool(default)
    if not isinstance(value, bool):
        raise CaptureConfigError(f"{name} must be a boolean, not {type(value).__name__}")
    return value

def _resolve_path(value: str | os.PathLike[str], *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_pipeline_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CaptureConfigError(f"config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise CaptureConfigError(f"invalid JSON config {config_path}: {exc}") from exc
    config = validate_pipeline_config(payload)
    config["__config_path"] = str(config_path)
    config["__config_dir"] = str(config_path.parent)
    config["output_dir"] = str(_resolve_path(config["output_dir"], base=config_path.parent))
    model_path = config["capture"].get("model_path")
    if model_path:
        config["capture"]["model_path"] = str(_resolve_path(model_path, base=config_path.parent))
    fate_path = config.get("prediction", {}).get("fate_artifact_path")
    if fate_path:
        config["prediction"]["fate_artifact_path"] = str(_resolve_path(fate_path, base=config_path.parent))
    return config


def validate_pipeline_config(payload: Any) -> dict[str, Any]:
    config = deepcopy(_require_mapping(payload, "config"))
    _reject_unknown(config, {
        "schema_version", "output_dir", "capture", "dataset", "payload",
        "prediction", "fixture", "simulation", "launcher", "model_runner",
        "multinode", "__config_path", "__config_dir",
    }, "config")
    schema = str(config.get("schema_version", SCHEMA_VERSION))
    if schema != SCHEMA_VERSION:
        raise CaptureConfigError(f"schema_version must be {SCHEMA_VERSION}")
    config["schema_version"] = schema
    config["output_dir"] = _require_nonempty(config.get("output_dir"), "output_dir")

    capture = _require_mapping(config.get("capture"), "capture")
    _reject_unknown(capture, {
        "backend", "capture_id", "request_id", "model_id", "model_path",
        "collector_version", "strict", "capture_compute",
        "infer_padding_from_zero_prob", "layer_id_offset", "sample_id_prefix",
        "minimum_consecutive_layers", "expected_moe_layer_count",
        "expected_layer_ids", "require_all_expected_layers",
        "global_rank_to_source_rank", "rank_to_node", "expert_to_rank",
        "accepted_source_ranks", "performance_eligible",
    }, "capture")
    backend = str(capture.get("backend", "MEGATRON_CORE_AUTO")).upper()
    if backend not in SUPPORTED_BACKENDS:
        raise CaptureConfigError(f"capture.backend must be one of {sorted(SUPPORTED_BACKENDS)}")
    capture["backend"] = backend
    capture["capture_id"] = _require_nonempty(capture.get("capture_id"), "capture.capture_id")
    capture["request_id"] = _require_nonempty(capture.get("request_id", capture["capture_id"]), "capture.request_id")
    capture["model_id"] = _require_nonempty(capture.get("model_id"), "capture.model_id")
    capture["collector_version"] = str(capture.get("collector_version", "current-p12-capture"))
    capture["strict"] = _require_bool(capture.get("strict"), "capture.strict", default=True)
    capture["capture_compute"] = _require_bool(capture.get("capture_compute"), "capture.capture_compute", default=True)
    capture["infer_padding_from_zero_prob"] = _require_bool(capture.get("infer_padding_from_zero_prob"), "capture.infer_padding_from_zero_prob", default=False)
    capture["layer_id_offset"] = int(capture.get("layer_id_offset", 0))
    capture["sample_id_prefix"] = str(capture.get("sample_id_prefix", capture["request_id"]))
    capture["minimum_consecutive_layers"] = int(capture.get("minimum_consecutive_layers", 2))
    if capture["minimum_consecutive_layers"] < 2:
        raise CaptureConfigError("capture.minimum_consecutive_layers must be at least 2")
    expected_count = capture.get("expected_moe_layer_count")
    capture["expected_moe_layer_count"] = None if expected_count in (None, "") else int(expected_count)
    if capture["expected_moe_layer_count"] is not None and capture["expected_moe_layer_count"] < 2:
        raise CaptureConfigError("capture.expected_moe_layer_count must be at least 2")
    expected_ids = capture.get("expected_layer_ids")
    if expected_ids is not None:
        normalized_ids = sorted({int(value) for value in expected_ids})
        if len(normalized_ids) < 2 or any(value < 0 for value in normalized_ids):
            raise CaptureConfigError("capture.expected_layer_ids must contain at least two nonnegative layer IDs")
        if any(normalized_ids[index + 1] != normalized_ids[index] + 1 for index in range(len(normalized_ids) - 1)):
            raise CaptureConfigError("capture.expected_layer_ids must be consecutive for Current P12")
        capture["expected_layer_ids"] = normalized_ids
        if capture["expected_moe_layer_count"] is not None and len(normalized_ids) != capture["expected_moe_layer_count"]:
            raise CaptureConfigError("expected_layer_ids length differs from expected_moe_layer_count")
    else:
        capture["expected_layer_ids"] = None
    capture["require_all_expected_layers"] = _require_bool(
        capture.get("require_all_expected_layers"),
        "capture.require_all_expected_layers",
        default=(capture["expected_moe_layer_count"] is not None or capture["expected_layer_ids"] is not None),
    )
    capture["global_rank_to_source_rank"] = {
        str(key): int(value) for key, value in dict(capture.get("global_rank_to_source_rank", {})).items()
    }
    for name in ("rank_to_node", "expert_to_rank", "accepted_source_ranks"):
        if capture.get(name) is not None:
            values = tuple(int(v) for v in capture[name])
            if not values or any(v < 0 for v in values):
                raise CaptureConfigError(f"capture.{name} must contain nonnegative integers")
            if name == "accepted_source_ranks" and len(set(values)) != len(values):
                raise CaptureConfigError("capture.accepted_source_ranks must not contain duplicates")
            capture[name] = list(values)
    config["capture"] = capture

    dataset = _require_mapping(config.get("dataset"), "dataset")
    _reject_unknown(dataset, {
        "dataset_id", "split", "source_kind", "notes", "input_contract",
        "seq_length", "local_micro_batch_size", "global_source_batch_size",
        "local_input_tokens", "global_input_tokens", "seed", "save_token_ids",
        "qualification_samples", "qualification_threshold_ratio",
    }, "dataset")
    dataset["dataset_id"] = _require_nonempty(dataset.get("dataset_id"), "dataset.dataset_id")
    dataset["split"] = str(dataset.get("split", "test"))
    if dataset["split"] not in SUPPORTED_SPLITS:
        raise CaptureConfigError(f"dataset.split must be one of {sorted(SUPPORTED_SPLITS)}")
    dataset["source_kind"] = str(dataset.get("source_kind", "megatron_runtime_capture"))
    dataset["notes"] = str(dataset.get("notes", ""))
    config["dataset"] = dataset

    payload = _require_mapping(config.get("payload"), "payload")
    _reject_unknown(payload, {"dispatch", "combine", "descriptor"}, "payload")
    for phase in ("dispatch", "combine"):
        spec = _require_mapping(payload.get(phase), f"payload.{phase}")
        _reject_unknown(spec, {
            "token_payload_bytes_per_row", "auxiliary_payload_bytes_per_row",
            "metadata_bytes_per_edge", "alignment_bytes", "padding_rule", "dtype",
        }, f"payload.{phase}")
        for name in (
            "token_payload_bytes_per_row",
            "auxiliary_payload_bytes_per_row",
            "metadata_bytes_per_edge",
        ):
            spec[name] = int(spec.get(name, 0))
            if spec[name] < 0:
                raise CaptureConfigError(f"payload.{phase}.{name} must be nonnegative")
        spec["alignment_bytes"] = int(spec.get("alignment_bytes", 1))
        if spec["alignment_bytes"] <= 0:
            raise CaptureConfigError(f"payload.{phase}.alignment_bytes must be positive")
        spec["padding_rule"] = str(spec.get("padding_rule", "NONE"))
        if spec["padding_rule"] not in {"NONE", "EDGE_TOTAL_ALIGN_UP"}:
            raise CaptureConfigError(f"payload.{phase}.padding_rule must be NONE or EDGE_TOTAL_ALIGN_UP")
        spec["dtype"] = _require_nonempty(spec.get("dtype"), f"payload.{phase}.dtype")
        payload[phase] = spec
    descriptor = _require_mapping(payload.get("descriptor"), "payload.descriptor")
    _reject_unknown(descriptor, {
        "fixed_header_bytes", "per_destination_entry_bytes",
    }, "payload.descriptor")
    descriptor["fixed_header_bytes"] = int(descriptor.get("fixed_header_bytes", 0))
    descriptor["per_destination_entry_bytes"] = int(descriptor.get("per_destination_entry_bytes", 0))
    if descriptor["fixed_header_bytes"] < 0 or descriptor["per_destination_entry_bytes"] < 0:
        raise CaptureConfigError("descriptor byte costs must be nonnegative")
    payload["descriptor"] = descriptor
    config["payload"] = payload

    prediction = _require_mapping(config.get("prediction", {}), "prediction")
    _reject_unknown(prediction, {
        "mode", "fate_artifact_path", "provider", "max_sample_tokens",
        "confidence_ppm", "require_complete_fate_coverage",
    }, "prediction")
    prediction["mode"] = str(prediction.get("mode", "FATE_P2")).upper()
    if prediction["mode"] == "FATE":
        prediction["mode"] = "FATE_P2"
    if prediction["mode"] != "FATE_P2":
        raise CaptureConfigError("prediction.mode must be FATE_P2")
    fate_path = prediction.get("fate_artifact_path")
    prediction["fate_artifact_path"] = None if fate_path in (None, "") else str(fate_path)
    default_provider = "EXTERNAL_ARTIFACT" if prediction["fate_artifact_path"] else "MEGATRON_SAMPLED_FATE"
    prediction["provider"] = str(prediction.get("provider", default_provider)).upper()
    if prediction["provider"] not in {"EXTERNAL_ARTIFACT", "MEGATRON_SAMPLED_FATE"}:
        raise CaptureConfigError("prediction.provider must be EXTERNAL_ARTIFACT or MEGATRON_SAMPLED_FATE")
    if prediction["provider"] == "EXTERNAL_ARTIFACT" and prediction["fate_artifact_path"] is None:
        raise CaptureConfigError("prediction.fate_artifact_path is required for EXTERNAL_ARTIFACT")
    prediction["max_sample_tokens"] = int(prediction.get("max_sample_tokens", 2048))
    if prediction["max_sample_tokens"] <= 0:
        raise CaptureConfigError("prediction.max_sample_tokens must be positive")
    prediction["confidence_ppm"] = int(prediction.get("confidence_ppm", 750000))
    if not 0 <= prediction["confidence_ppm"] <= 1_000_000:
        raise CaptureConfigError("prediction.confidence_ppm must be in [0, 1000000]")
    prediction["require_complete_fate_coverage"] = _require_bool(
        prediction.get("require_complete_fate_coverage"),
        "prediction.require_complete_fate_coverage",
        default=True,
    )
    config["prediction"] = prediction

    fixture = _require_mapping(config.get("fixture", {}), "fixture")
    _reject_unknown(fixture, {
        "fixture_id_prefix", "compute_fallback_ns", "allow_compute_fallback",
    }, "fixture")
    fixture["fixture_id_prefix"] = str(fixture.get("fixture_id_prefix", capture["capture_id"]))
    fixture["compute_fallback_ns"] = _validate_compute_fallback(fixture.get("compute_fallback_ns", {}))
    fixture["allow_compute_fallback"] = _require_bool(fixture.get("allow_compute_fallback"), "fixture.allow_compute_fallback", default=True)
    config["fixture"] = fixture

    simulation = _require_mapping(config.get("simulation", {}), "simulation")
    _reject_unknown(simulation, {
        "enabled", "algorithm", "information", "overlap", "release",
        "p0_p1_compute_end_barrier", "staging", "max_task_bytes",
        "max_timestamps", "policy", "planning", "scope", "safe_selector",
    }, "simulation")
    simulation["enabled"] = _require_bool(
        simulation.get("enabled"), "simulation.enabled", default=True
    )
    legacy = sorted(key for key in ("policy", "planning", "scope", "safe_selector") if key in simulation)
    if legacy:
        raise CaptureConfigError(
            "simulation must use one algorithm expression; remove legacy fields: " + ", ".join(legacy)
        )
    simulation.setdefault("algorithm", "joint(global_(rscf()))")
    simulation.setdefault("information", config["prediction"]["mode"])
    simulation.setdefault("overlap", "OVERLAP")
    simulation.setdefault("release", PAPER_RELEASE_MODE)
    simulation["p0_p1_compute_end_barrier"] = _require_bool(
        simulation.get("p0_p1_compute_end_barrier"),
        "simulation.p0_p1_compute_end_barrier",
        default=PAPER_P0_P1_COMPUTE_END_BARRIER,
    )
    simulation.setdefault("staging", "1.0X")
    simulation.setdefault("max_task_bytes", PAPER_MAX_TASK_BYTES)
    simulation.setdefault("max_timestamps", 100000)
    config["simulation"] = simulation

    launcher = _require_mapping(config.get("launcher", {}), "launcher")
    _reject_unknown(launcher, {
        "command", "environment", "timeout_seconds", "clean_output_before_collect",
    }, "launcher")
    command = launcher.get("command", [])
    if isinstance(command, str):
        raise CaptureConfigError("launcher.command must be a JSON string array, not a shell string")
    launcher["command"] = [str(item) for item in command]
    launcher["environment"] = {str(k): str(v) for k, v in dict(launcher.get("environment", {})).items()}
    launcher["timeout_seconds"] = int(launcher.get("timeout_seconds", 0))
    launcher["clean_output_before_collect"] = _require_bool(launcher.get("clean_output_before_collect"), "launcher.clean_output_before_collect", default=True)
    config["launcher"] = launcher
    return config


def _validate_compute_fallback(value: Any) -> dict[str, int]:
    supplied = dict(value or {})
    defaults = {
        "combine_release_to_router_ready_ns": 0,
        "router_and_pack_ns": 0,
        "dispatch_local_postprocess_ns": 0,
        "dispatch_release_to_combine_source_ready_ns": 0,
        "bootstrap_router_and_pack_ns": 0,
    }
    for key in defaults:
        defaults[key] = int(supplied.get(key, defaults[key]))
        if defaults[key] < 0:
            raise CaptureConfigError(f"fixture.compute_fallback_ns.{key} must be nonnegative")
    return defaults


def example_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "output_dir": "outputs/trace/olmoe_ep4",
        "capture": {
            "backend": "MEGATRON_CORE_AUTO",
            "capture_id": "olmoe-ep4-smoke",
            "request_id": "olmoe-smoke-request",
            "sample_id_prefix": "olmoe-smoke",
            "model_id": "OLMoE-1B-7B-0924",
            "model_path": "/models/OLMoE-1B-7B-0924",
            "strict": True,
            "capture_compute": True,
            "infer_padding_from_zero_prob": False,
            "rank_to_node": [0, 0, 0, 0],
            "expert_to_rank": None,
            "global_rank_to_source_rank": {},
        },
        "prediction": {
            "mode": "FATE_P2",
            "fate_artifact_path": None,
            "provider": "MEGATRON_SAMPLED_FATE",
            "max_sample_tokens": 2048,
            "confidence_ppm": 750000,
            "require_complete_fate_coverage": False,
        },
        "dataset": {
            "dataset_id": "olmoe-trace-smoke",
            "split": "test",
            "source_kind": "megatron_runtime_capture",
            "notes": "The self-contained AutoBridge runner launches the model; no external entrypoint is required.",
        },
        "payload": {
            "dispatch": {
                "token_payload_bytes_per_row": 4096,
                "auxiliary_payload_bytes_per_row": 16,
                "metadata_bytes_per_edge": 64,
                "alignment_bytes": 256,
                "padding_rule": "EDGE_TOTAL_ALIGN_UP",
                "dtype": "bfloat16",
            },
            "combine": {
                "token_payload_bytes_per_row": 4096,
                "auxiliary_payload_bytes_per_row": 8,
                "metadata_bytes_per_edge": 48,
                "alignment_bytes": 128,
                "padding_rule": "EDGE_TOTAL_ALIGN_UP",
                "dtype": "bfloat16",
            },
            "descriptor": {"fixed_header_bytes": 64, "per_destination_entry_bytes": 16},
        },
        "fixture": {
            "fixture_id_prefix": "olmoe-ep4",
            "allow_compute_fallback": True,
            "compute_fallback_ns": {
                "combine_release_to_router_ready_ns": 0,
                "router_and_pack_ns": 0,
                "dispatch_local_postprocess_ns": 0,
                "dispatch_release_to_combine_source_ready_ns": 0,
                "bootstrap_router_and_pack_ns": 0,
            },
        },
        "simulation": {
            "enabled": True,
            "algorithm": "joint(global_(rscf()))",
            "information": "FATE_P2",
            "overlap": "OVERLAP",
            "release": PAPER_RELEASE_MODE,
            "p0_p1_compute_end_barrier": PAPER_P0_P1_COMPUTE_END_BARRIER,
            "staging": "1.0X",
            "max_task_bytes": 1 << 18,
            "max_timestamps": 100000,
        },
        "launcher": {
            "command": [
                "torchrun",
                "--standalone",
                "--nproc_per_node=4",
                "YOUR_EXISTING_MEGATRON_MODEL_RUNNER.py",
            ],
            "environment": {},
            "timeout_seconds": 0,
            "clean_output_before_collect": True,
        },
    }
