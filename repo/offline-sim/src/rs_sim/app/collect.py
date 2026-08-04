from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from rs_sim.trace.collection.config import validate_pipeline_config
from rs_sim.trace.collection.fixture_builder import build_fixtures_from_capture
from rs_sim.trace.collection.pipeline import doctor, inspect_capture_artifacts, launch_collection
from rs_sim.trace.io.serialization import load_fixture
from rs_sim.trace.schema.validation import validate_fixture
from rs_sim.trace.runners.model_support import (
    ModelSupportError,
    inspect_hf_model,
    validate_generic_text_moe,
)
from rs_sim.trace.runners.preflight import run_megatron_model_preflight, write_preflight

from .artifact_identity import checkpoint_identity
from .config_io import (
    ConfigError, reject_unknown_fields, require_bool, require_mapping,
    require_nonempty, resolve_path,
)


def _dtype_bytes(dtype: str) -> int:
    mapping = {
        "float32": 4,
        "fp32": 4,
        "float16": 2,
        "fp16": 2,
        "bfloat16": 2,
        "bf16": 2,
        "float8": 1,
        "fp8": 1,
    }
    try:
        return mapping[str(dtype).lower()]
    except KeyError as exc:
        raise ConfigError(f"unsupported dtype for payload inference: {dtype}") from exc


def _launcher_command(
    config: dict[str, Any],
    model: dict[str, Any],
    launch: dict[str, Any],
    parallel: dict[str, Any],
    input_cfg: dict[str, Any],
    capture: dict[str, Any],
    *,
    model_path: Path,
) -> list[str]:
    explicit = launch.get("command")
    if explicit:
        if not isinstance(explicit, list):
            raise ConfigError("launch.command must be a list")
        return [str(item) for item in explicit]

    launcher = str(launch.get("launcher", "torchrun"))
    entrypoint = str(model.get("entrypoint", "")).strip()
    runner = str(model.get("runner", "AUTO_MEGATRON_BRIDGE")).upper()
    if entrypoint:
        if launcher == "torchrun":
            command = [
                "torchrun",
                f"--nnodes={int(launch.get('nnodes', 1))}",
                f"--nproc_per_node={int(launch.get('nproc_per_node', 1))}",
            ]
            if int(launch.get("nnodes", 1)) == 1 and not launch.get("master_addr"):
                command.append("--standalone")
            if launch.get("node_rank") is not None:
                command.append(f"--node_rank={int(launch['node_rank'])}")
            if launch.get("master_addr"):
                command.append(f"--master_addr={launch['master_addr']}")
            if launch.get("master_port"):
                command.append(f"--master_port={int(launch['master_port'])}")
            command.append(entrypoint)
        else:
            command = [launcher, entrypoint]
        command.extend(str(item) for item in model.get("args", []))
        command.extend(str(item) for item in launch.get("extra_args", []))
        return command

    if runner not in {"AUTO_MEGATRON_BRIDGE", "MEGATRON_BRIDGE_AUTO_TEXT_MOE"}:
        raise ConfigError(
            "model.entrypoint is absent and model.runner is not AUTO_MEGATRON_BRIDGE"
        )
    if launcher != "torchrun":
        raise ConfigError("the built-in Megatron Bridge runner requires launch.launcher=torchrun")

    ep = int(parallel.get("ep", 1))
    tp = int(parallel.get("tp", 1))
    pp = int(parallel.get("pp", 1))
    dp = int(parallel.get("dp", 1))
    cp = int(parallel.get("cp", 1))
    etp = int(parallel.get("etp", 1))
    nnodes = int(launch.get("nnodes", 1))
    nproc = int(launch.get("nproc_per_node", ep * tp * pp * dp))
    expected_world = ep * tp * pp * dp
    if nnodes * nproc != expected_world:
        raise ConfigError(
            f"launch world size {nnodes}*{nproc} differs from EP*TP*PP*DP={expected_world}"
        )

    command = [
        "torchrun",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc}",
    ]
    if nnodes == 1 and not launch.get("master_addr"):
        command.append("--standalone")
    if launch.get("node_rank") is not None:
        command.append(f"--node_rank={int(launch['node_rank'])}")
    if launch.get("master_addr"):
        command.append(f"--master_addr={launch['master_addr']}")
    if launch.get("master_port"):
        command.append(f"--master_port={int(launch['master_port'])}")
    command.extend(["--module", "rs_sim.trace.runners.megatron_bridge_moe"])
    command.extend(["--model-path", str(model_path)])
    command.extend(["--model-format", str(model.get("format", "auto")).lower()])
    hf_config_path = model.get("hf_config_path")
    if hf_config_path:
        command.extend(["--hf-config-path", str(resolve_path(str(hf_config_path), config=config))])
    micro_batch_size = int(input_cfg.get("micro_batch_size", 1))
    global_batch_size = int(input_cfg.get("global_batch_size", micro_batch_size * ep))
    command.extend([
        "--tp", str(tp),
        "--pp", str(pp),
        "--ep", str(ep),
        "--etp", str(etp),
        "--cp", str(cp),
        "--dp", str(dp),
        "--dtype", str(model.get("dtype", "bfloat16")),
        "--dispatcher", str(model.get("dispatcher", "alltoall")),
        "--seq-length", str(int(input_cfg.get("seq_length", 128))),
        "--micro-batch-size", str(micro_batch_size),
        "--global-batch-size", str(global_batch_size),
        "--warmup-samples", str(int(input_cfg.get("warmup_count", input_cfg.get("warmup", 1)))),
        "--samples", str(int(input_cfg.get("sample_count", input_cfg.get("samples", 2)))),
        "--seed", str(int(input_cfg.get("seed", 1234))),
        "--qualification-samples", str(int(input_cfg.get("qualification_samples", 0))),
        "--qualification-threshold-ratio", str(float(input_cfg.get("qualification_threshold_ratio", 0.05))),
    ])
    if require_bool(input_cfg.get("save_token_ids"), "input.save_token_ids", default=True):
        command.append("--save-input-ids")
    if require_bool(model.get("trust_remote_code"), "model.trust_remote_code", default=True):
        command.append("--trust-remote-code")
    if require_bool(model.get("random_init"), "model.random_init", default=False):
        command.append("--random-init")
    command.extend(str(item) for item in model.get("args", []))
    command.extend(str(item) for item in launch.get("extra_args", []))
    return command


def normalize_collect_config(config: dict[str, Any]) -> dict[str, Any]:
    reject_unknown_fields(config, {
        "version", "name", "model", "launch", "parallel", "input", "capture",
        "prediction", "payload", "output", "multinode", "__config_path", "__config_dir",
    }, "collect config")
    version = int(config.get("version", 1))
    if version != 1:
        raise ConfigError("collect config version must be 1")
    name = require_nonempty(config.get("name", "trace-collection"), "name")
    model = require_mapping(config.get("model"), "model")
    launch = require_mapping(config.get("launch", {}), "launch")
    parallel = require_mapping(config.get("parallel"), "parallel")
    capture = require_mapping(config.get("capture", {}), "capture")
    input_cfg = require_mapping(config.get("input", {}), "input")
    output_cfg = require_mapping(config.get("output"), "output")
    multinode_cfg = require_mapping(config.get("multinode", {}), "multinode")
    reject_unknown_fields(model, {
        "framework", "runner", "family", "id", "path", "format", "dtype",
        "dispatcher", "trust_remote_code", "random_init", "hidden_size",
        "entrypoint", "args", "hf_config_path",
    }, "model")
    reject_unknown_fields(launch, {
        "launcher", "command", "nnodes", "nproc_per_node", "timeout_seconds",
        "environment", "extra_args", "master_addr", "master_port", "node_rank",
    }, "launch")
    reject_unknown_fields(parallel, {"ep", "tp", "pp", "dp", "cp", "etp"}, "parallel")
    reject_unknown_fields(input_cfg, {
        "dataset_id", "dataset", "split", "seq_length", "micro_batch_size",
        "global_batch_size", "warmup_count", "warmup", "sample_count", "samples",
        "seed", "notes", "save_token_ids", "qualification_samples",
        "qualification_threshold_ratio",
    }, "input")
    reject_unknown_fields(capture, {
        "backend", "capture_id", "request_id", "sample_id_prefix", "strict",
        "local_compute", "allow_compute_fallback", "infer_padding_from_zero_prob",
        "layer_id_offset", "minimum_consecutive_layers", "expected_moe_layer_count",
        "expected_layer_ids", "require_all_expected_layers", "rank_to_node",
        "expert_to_rank", "global_rank_to_source_rank", "accepted_source_ranks",
        "compute_fallback_ns", "fixture_id_prefix",
    }, "capture")
    reject_unknown_fields(output_cfg, {"directory", "overwrite"}, "output")
    reject_unknown_fields(multinode_cfg, {
        "artifact_owner_node_rank", "node_artifact_transport",
    }, "multinode")

    output_dir = resolve_path(require_nonempty(output_cfg.get("directory"), "output.directory"), config=config)
    model_path = resolve_path(require_nonempty(model.get("path"), "model.path"), config=config)
    ep = int(parallel.get("ep", 1))
    tp = int(parallel.get("tp", 1))
    pp = int(parallel.get("pp", 1))
    dp = int(parallel.get("dp", 1))
    cp = int(parallel.get("cp", 1))
    etp = int(parallel.get("etp", 1))
    if min(ep, tp, pp, dp, cp, etp) <= 0:
        raise ConfigError("parallel dimensions must be positive")
    built_in_runner = not launch.get("command") and not str(model.get("entrypoint", "")).strip()
    seq_length = int(input_cfg.get("seq_length", 128))
    micro_batch_size = int(input_cfg.get("micro_batch_size", 1))
    global_batch_size = int(input_cfg.get("global_batch_size", micro_batch_size * ep))
    if min(seq_length, micro_batch_size, global_batch_size) <= 0:
        raise ConfigError("input sequence length and batch sizes must be positive")
    if built_in_runner and global_batch_size != micro_batch_size * ep:
        raise ConfigError(
            "input.global_batch_size must equal input.micro_batch_size * parallel.ep "
            "for the EP source-rank trace contract"
        )
    qualification_samples = int(input_cfg.get("qualification_samples", 0))
    qualification_threshold_ratio = float(input_cfg.get("qualification_threshold_ratio", 0.05))
    measured_samples = int(input_cfg.get("sample_count", input_cfg.get("samples", 2)))
    if qualification_samples < 0:
        raise ConfigError("input.qualification_samples must be nonnegative")
    if qualification_samples > measured_samples:
        raise ConfigError("input.qualification_samples must not exceed input.sample_count")
    if not 0.0 <= qualification_threshold_ratio <= 1.0:
        raise ConfigError("input.qualification_threshold_ratio must be in [0,1]")

    inspection = None
    if built_in_runner:
        # The current trace contract identifies sources by EP rank.  Reject
        # mixed model parallelism rather than emit duplicate/ambiguous rows.
        if tp != 1 or pp != 1 or dp != 1 or cp != 1:
            raise ConfigError(
                "built-in Current-P12 collection currently requires TP=PP=DP=CP=1; "
                "model-family support is automatic, mixed-rank projection is fail-closed"
            )
        model_format = str(model.get("format", "auto")).lower()
        inspection_path = model_path
        if model_format == "megatron" or (
            model_format == "auto" and not (model_path / "config.json").is_file()
        ):
            hf_config_path = model.get("hf_config_path")
            if not hf_config_path:
                raise ConfigError(
                    "native Megatron checkpoints require model.hf_config_path for AutoBridge architecture detection"
                )
            inspection_path = resolve_path(str(hf_config_path), config=config)
        try:
            inspection = inspect_hf_model(inspection_path)
            validate_generic_text_moe(inspection, ep=ep)
        except ModelSupportError as exc:
            raise ConfigError(str(exc)) from exc

    rank_to_node = capture.get("rank_to_node")
    if rank_to_node is None:
        gpus_per_node = int(launch.get("nproc_per_node", ep))
        rank_to_node = [index // max(1, gpus_per_node) for index in range(ep)]
    rank_to_node = [int(item) for item in rank_to_node]
    if len(rank_to_node) != ep:
        raise ConfigError("capture.rank_to_node length must equal parallel.ep")

    dtype = str(model.get("dtype", "bfloat16"))
    hidden_size = int(model.get("hidden_size", 0))
    if hidden_size <= 0 and inspection is not None and inspection.hidden_size is not None:
        hidden_size = int(inspection.hidden_size)
    payload = require_mapping(config.get("payload", {}), "payload")
    prediction = require_mapping(config.get("prediction", {}), "prediction")
    reject_unknown_fields(payload, {"dispatch", "combine", "descriptor"}, "payload")
    reject_unknown_fields(prediction, {
        "mode", "provider", "fate_artifact_path", "max_sample_tokens",
        "confidence_ppm", "require_complete_fate_coverage",
    }, "prediction")
    inferred_row_bytes = hidden_size * _dtype_bytes(dtype) if hidden_size > 0 else 0

    def phase_payload(name: str, *, default_aux: int, default_header: int, default_alignment: int) -> dict[str, Any]:
        supplied = require_mapping(payload.get(name, {}), f"payload.{name}")
        reject_unknown_fields(supplied, {
            "token_payload_bytes_per_row", "auxiliary_payload_bytes_per_row",
            "metadata_bytes_per_edge", "alignment_bytes", "padding_rule", "dtype",
        }, f"payload.{name}")
        row_bytes = int(supplied.get("token_payload_bytes_per_row", inferred_row_bytes))
        if row_bytes <= 0:
            raise ConfigError(
                f"payload.{name}.token_payload_bytes_per_row is required when model.hidden_size is absent"
            )
        return {
            "token_payload_bytes_per_row": row_bytes,
            "auxiliary_payload_bytes_per_row": int(supplied.get("auxiliary_payload_bytes_per_row", default_aux)),
            "metadata_bytes_per_edge": int(supplied.get("metadata_bytes_per_edge", default_header)),
            "alignment_bytes": int(supplied.get("alignment_bytes", default_alignment)),
            "padding_rule": str(supplied.get("padding_rule", "EDGE_TOTAL_ALIGN_UP")),
            "dtype": str(supplied.get("dtype", dtype)),
        }

    descriptor = require_mapping(payload.get("descriptor", {}), "payload.descriptor")
    reject_unknown_fields(descriptor, {
        "fixed_header_bytes", "per_destination_entry_bytes",
    }, "payload.descriptor")
    compute = require_mapping(capture.get("compute_fallback_ns", {}), "capture.compute_fallback_ns")
    reject_unknown_fields(compute, {
        "combine_release_to_router_ready_ns", "router_and_pack_ns",
        "dispatch_local_postprocess_ns",
        "dispatch_release_to_combine_source_ready_ns",
        "bootstrap_router_and_pack_ns",
    }, "capture.compute_fallback_ns")
    fallback = {
        "combine_release_to_router_ready_ns": int(compute.get("combine_release_to_router_ready_ns", 0)),
        "router_and_pack_ns": int(compute.get("router_and_pack_ns", 0)),
        "dispatch_local_postprocess_ns": int(compute.get("dispatch_local_postprocess_ns", 0)),
        "dispatch_release_to_combine_source_ready_ns": int(compute.get("dispatch_release_to_combine_source_ready_ns", 0)),
        "bootstrap_router_and_pack_ns": int(compute.get("bootstrap_router_and_pack_ns", 0)),
    }

    internal = {
        "schema_version": "RS_SIM_TRACE_CAPTURE_PIPELINE",
        "output_dir": str(output_dir),
        "capture": {
            "backend": str(capture.get("backend", "MEGATRON_CORE_AUTO")),
            "capture_id": str(capture.get("capture_id", name)),
            "request_id": str(capture.get("request_id", name)),
            "sample_id_prefix": str(capture.get("sample_id_prefix", name)),
            "model_id": str(model.get("family", model.get("id", name))),
            "model_path": str(model_path),
            "collector_version": "current-p12-self-contained-megatron",
            "strict": require_bool(capture.get("strict"), "capture.strict", default=True),
            "capture_compute": require_bool(capture.get("local_compute"), "capture.local_compute", default=True),
            "infer_padding_from_zero_prob": require_bool(capture.get("infer_padding_from_zero_prob"), "capture.infer_padding_from_zero_prob", default=False),
            "layer_id_offset": int(capture.get("layer_id_offset", 0)),
            "minimum_consecutive_layers": int(capture.get("minimum_consecutive_layers", 2)),
            "expected_moe_layer_count": (
                None if capture.get("expected_moe_layer_count") in (None, "")
                else int(capture.get("expected_moe_layer_count"))
            ),
            "expected_layer_ids": capture.get("expected_layer_ids"),
            "require_all_expected_layers": require_bool(capture.get("require_all_expected_layers"), "capture.require_all_expected_layers", default=True),
            "rank_to_node": rank_to_node,
            "expert_to_rank": capture.get("expert_to_rank"),
            "global_rank_to_source_rank": capture.get("global_rank_to_source_rank", {}),
            "accepted_source_ranks": capture.get("accepted_source_ranks"),
        },
        "dataset": {
            "dataset_id": str(input_cfg.get("dataset_id", input_cfg.get("dataset", name))),
            "split": str(input_cfg.get("split", "test")),
            "source_kind": "megatron_runtime_capture",
            "notes": str(input_cfg.get("notes", "")),
            "input_contract": (
                "COUNTER_SEEDED_GLOBAL_SAMPLE" if built_in_runner else "EXTERNAL_RUNNER_UNVERIFIED"
            ),
            "seq_length": seq_length,
            "local_micro_batch_size": micro_batch_size,
            "global_source_batch_size": global_batch_size if built_in_runner else None,
            "local_input_tokens": seq_length * micro_batch_size,
            "global_input_tokens": seq_length * global_batch_size if built_in_runner else None,
            "seed": int(input_cfg.get("seed", 1234)),
            "save_token_ids": require_bool(input_cfg.get("save_token_ids"), "input.save_token_ids", default=True) if built_in_runner else False,
            "qualification_samples": qualification_samples if built_in_runner else 0,
            "qualification_threshold_ratio": qualification_threshold_ratio,
        },
        "payload": {
            "dispatch": phase_payload("dispatch", default_aux=0, default_header=0, default_alignment=1),
            "combine": phase_payload("combine", default_aux=0, default_header=0, default_alignment=1),
            "descriptor": {
                "fixed_header_bytes": int(descriptor.get("fixed_header_bytes", 0)),
                "per_destination_entry_bytes": int(descriptor.get("per_destination_entry_bytes", 0)),
            },
        },
        "prediction": {
            "mode": str(prediction.get("mode", "FATE_P2")),
            "provider": str(prediction.get("provider", "MEGATRON_SAMPLED_FATE")),
            "fate_artifact_path": prediction.get("fate_artifact_path"),
            "max_sample_tokens": int(prediction.get("max_sample_tokens", 2048)),
            "confidence_ppm": int(prediction.get("confidence_ppm", 750000)),
            "require_complete_fate_coverage": require_bool(
                prediction.get("require_complete_fate_coverage"),
                "prediction.require_complete_fate_coverage",
                default=True,
            ),
        },
        "fixture": {
            "fixture_id_prefix": str(capture.get("fixture_id_prefix", name)),
            "compute_fallback_ns": fallback,
            "allow_compute_fallback": require_bool(capture.get("allow_compute_fallback"), "capture.allow_compute_fallback", default=False),
        },
        "model_runner": {
            "kind": "LEGACY_EXTERNAL" if not built_in_runner else "MEGATRON_BRIDGE_AUTO_TEXT_MOE",
            "self_contained": bool(built_in_runner),
            "model_format": str(model.get("format", "auto")).upper(),
            "inspection": None if inspection is None else inspection.to_dict(),
            "parallel": {"ep": ep, "tp": tp, "pp": pp, "dp": dp, "cp": cp, "etp": etp},
        },
        "simulation": {"enabled": False},
        "multinode": {
            "enabled": int(launch.get("nnodes", 1)) > 1,
            "artifact_owner_node_rank": int(multinode_cfg.get("artifact_owner_node_rank", 0)),
            "node_artifact_transport": str(multinode_cfg.get("node_artifact_transport", "EXTERNAL_TRANSFER")),
        },
        "launcher": {
            "command": _launcher_command(
                config,
                model,
                launch,
                parallel,
                input_cfg,
                capture,
                model_path=model_path,
            ),
            "environment": {str(k): str(v) for k, v in dict(launch.get("environment", {})).items()},
            "timeout_seconds": int(launch.get("timeout_seconds", 0)),
            "clean_output_before_collect": require_bool(output_cfg.get("overwrite"), "output.overwrite", default=False),
        },
    }
    internal = validate_pipeline_config(internal)
    internal["__config_path"] = str(config["__config_path"])
    internal["__config_dir"] = str(config["__config_dir"])
    return internal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_trace_manifest(config: dict[str, Any], fixture_paths: tuple[Path, ...]) -> Path:
    output_dir = Path(config["output_dir"])
    fixtures = []
    for path in fixture_paths:
        fixture = load_fixture(path)
        validation = validate_fixture(fixture)
        fixtures.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "fixture_id": fixture.fixture_id,
                "truth_digest": fixture.truth_digest(),
                "world_size": fixture.world_size,
                "window_count": len(fixture.windows),
                "validation_status": validation["status"],
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": "RS_SIM_TRACE_MANIFEST",
        "status": "PASS",
        "capture_id": config["capture"]["capture_id"],
        "model_id": config["capture"]["model_id"],
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
        "expected_moe_layer_count": config["capture"].get("expected_moe_layer_count"),
        "expected_layer_ids": config["capture"].get("expected_layer_ids"),
        "all_expected_layers_required": bool(config["capture"].get("require_all_expected_layers", False)),
        "performance_eligible": False,
    }
    path = output_dir / "trace_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _bundle_trace(output_dir: Path, capture_id: str) -> tuple[Path, Path]:
    manifest_lines = []
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or "bundles" in path.relative_to(output_dir).parts:
            continue
        if path.name == "ARTIFACT_MANIFEST.sha256":
            continue
        files.append(path)
        manifest_lines.append(f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}\n")
    artifact_manifest = output_dir / "ARTIFACT_MANIFEST.sha256"
    artifact_manifest.write_text("".join(manifest_lines), encoding="utf-8")
    files.append(artifact_manifest)
    bundle_dir = output_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in capture_id)
    bundle = bundle_dir / f"{safe}_trace_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(output_dir).as_posix())
    sha_path = bundle.with_suffix(bundle.suffix + ".sha256")
    sha_path.write_text(f"{_sha256(bundle)}  {bundle.name}\n", encoding="utf-8")
    return bundle, sha_path


def _aggregate_runner_qualification(output_dir: Path, expected_world_size: int) -> dict[str, Any]:
    reports = []
    for path in sorted((Path(output_dir) / "runner").glob("rank*_megatron_runner.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        qualification = payload.get("capture_qualification")
        if qualification:
            reports.append({"rank": payload.get("rank"), **qualification})
    if len(reports) != int(expected_world_size):
        return {
            "status": "NOT_QUALIFIED",
            "performance_eligible": False,
            "reason": f"qualification reports={len(reports)}, expected={expected_world_size}",
            "rank_reports": reports,
        }
    max_overhead = max(float(item.get("global_max_overhead_ratio", 1.0)) for item in reports)
    threshold = min(float(item.get("threshold_ratio", 0.05)) for item in reports)
    return {
        "status": "PASS" if max_overhead <= threshold else "FAILED",
        "performance_eligible": bool(max_overhead <= threshold),
        "global_max_overhead_ratio": max_overhead,
        "threshold_ratio": threshold,
        "rank_reports": reports,
    }


def run_collect(config: dict[str, Any]) -> dict[str, Any]:
    internal = normalize_collect_config(config)
    output_dir = Path(internal["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(internal.get("multinode", {}).get("enabled", False)):
        raise ConfigError(
            "multi-node collection requires --phase prepare/worker/finalize; "
            "the all-in-one phase is intentionally single-node only"
        )
    checkpoint = checkpoint_identity(
        Path(internal["capture"]["model_path"]),
        explicit_digest=config.get("model", {}).get("checkpoint_digest"),
    )
    (output_dir / "checkpoint_identity.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The injected child process consumes the internal capture schema, not the
    # user-facing YAML.  Persist that resolved JSON and point the bootstrap at it.
    runtime_config_path = output_dir / "capture_runtime_config.json"
    runtime_config_path.write_text(
        json.dumps(
            {key: value for key, value in internal.items() if not str(key).startswith("__")},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    internal["__config_path"] = str(runtime_config_path)
    internal["__config_dir"] = str(output_dir)

    preflight_report = None
    if internal.get("model_runner", {}).get("self_contained"):
        parallel = internal["model_runner"]["parallel"]
        command = internal["launcher"]["command"]
        try:
            nproc_arg = next(item for item in command if item.startswith("--nproc_per_node="))
            nproc_per_node = int(nproc_arg.split("=", 1)[1])
            preflight_report = run_megatron_model_preflight(
                model_path=internal["capture"]["model_path"],
                hf_config_path=next(
                    (command[index + 1] for index, item in enumerate(command[:-1]) if item == "--hf-config-path"),
                    None,
                ),
                model_format=internal["model_runner"]["model_format"],
                ep=parallel["ep"],
                tp=parallel["tp"],
                pp=parallel["pp"],
                dp=parallel["dp"],
                cp=parallel["cp"],
                etp=parallel["etp"],
                nproc_per_node=nproc_per_node,
                trust_remote_code="--trust-remote-code" in command,
                require_cuda=True,
                require_fate_route=internal["prediction"]["mode"] == "FATE_P2",
                require_compute_hooks=bool(internal["capture"].get("capture_compute", True)),
            )
            write_preflight(output_dir / "model_preflight.json", preflight_report)
        except BaseException as exc:
            failed = {
                "schema_version": "RS_SIM_MEGATRON_MODEL_PREFLIGHT",
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_runner": internal.get("model_runner"),
            }
            write_preflight(output_dir / "model_preflight.json", failed)
            raise RuntimeError(
                f"self-contained Megatron model preflight failed; see {output_dir / 'model_preflight.json'}: {exc}"
            ) from exc

    environment_report = doctor(
        internal,
        require_megatron=internal["capture"]["backend"] == "MEGATRON_CORE_AUTO",
    )
    launch_result = launch_collection(internal)
    artifact_status = inspect_capture_artifacts(internal)
    if artifact_status.get("status") != "PASS":
        raise RuntimeError(f"capture artifact validation failed: {artifact_status}")
    parallel = internal["model_runner"]["parallel"]
    expected_world = int(parallel["ep"] * parallel["tp"] * parallel["pp"] * parallel["dp"])
    qualification = _aggregate_runner_qualification(output_dir, expected_world)
    internal["capture"]["performance_eligible"] = bool(qualification.get("performance_eligible", False))
    internal["capture"]["performance_qualification"] = qualification
    fixtures = build_fixtures_from_capture(internal)
    trace_manifest = _write_trace_manifest(internal, fixtures)
    trace_payload = json.loads(trace_manifest.read_text(encoding="utf-8"))
    trace_payload["performance_qualification"] = qualification
    trace_payload["performance_eligible"] = bool(qualification.get("performance_eligible", False))
    trace_payload["checkpoint_identity"] = checkpoint
    trace_payload["input_contract"] = internal.get("dataset", {})
    trace_manifest.write_text(
        json.dumps(trace_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bundle, bundle_sha = _bundle_trace(output_dir, internal["capture"]["capture_id"])
    summary = {
        "schema_version": "RS_SIM_COLLECT_RESULT",
        "status": "PASS",
        "capture_id": internal["capture"]["capture_id"],
        "trace_manifest": str(trace_manifest),
        "fixture_paths": [str(path) for path in fixtures],
        "artifact_bundle": str(bundle),
        "artifact_bundle_sha256_file": str(bundle_sha),
        "environment": environment_report,
        "launch": launch_result,
        "capture_artifacts": artifact_status,
        "model_preflight": preflight_report,
        "checkpoint_identity": checkpoint,
        "performance_qualification": qualification,
        "input_contract": internal.get("dataset", {}),
    }
    (output_dir / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
