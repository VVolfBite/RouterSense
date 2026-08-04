"""One-command model capture → local fixture → formal simulation pipeline."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import zipfile
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from rs_sim import stable_digest
from rs_sim.runtime import build_current_p12_integration_runtime
from rs_sim.contracts.paper_defaults import PAPER_P0_P1_COMPUTE_END_BARRIER
from rs_sim.contracts.digest import stable_json_dumps
from rs_sim.trace.io.serialization import load_fixture
from rs_sim.trace.schema.validation import validate_fixture

from .fixture_builder import build_fixtures_from_capture


def doctor(config: dict[str, Any], *, require_megatron: bool = False) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    checks: dict[str, Any] = {
        "output_writable": True,
        "backend": config["capture"]["backend"],
        "model_path_exists": None,
        "torch_importable": None,
        "megatron_importable": None,
    }
    model_path = config["capture"].get("model_path")
    if model_path:
        checks["model_path_exists"] = Path(model_path).exists()
        if not checks["model_path_exists"] and config["capture"].get("strict", True):
            raise FileNotFoundError(f"model_path does not exist: {model_path}")
    if require_megatron or config["capture"]["backend"] == "MEGATRON_CORE_AUTO":
        try:
            import torch  # type: ignore
            checks["torch_importable"] = True
            checks["cuda_available"] = bool(torch.cuda.is_available())
        except Exception as exc:
            checks["torch_importable"] = False
            checks["torch_error"] = str(exc)
            if require_megatron:
                raise
        try:
            import megatron.core.transformer.moe.moe_layer  # type: ignore  # noqa: F401
            checks["megatron_importable"] = True
        except Exception as exc:
            checks["megatron_importable"] = False
            checks["megatron_error"] = str(exc)
            if require_megatron:
                raise
    return {"status": "PASS", "checks": checks}


def _bootstrap_dir(output_dir: Path) -> Path:
    path = output_dir / ".capture_bootstrap"
    path.mkdir(parents=True, exist_ok=True)
    (path / "sitecustomize.py").write_text(
        "from rs_sim.trace.collection.bootstrap import install_from_environment\ninstall_from_environment()\n",
        encoding="utf-8",
    )
    return path



def _clean_generated_output(output_dir: Path) -> None:
    for name in ("raw", "fixtures", "simulation", "bundles", ".capture_bootstrap"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in (
        "collect.log",
        "collect_result.json",
        "finalize_summary.json",
        "pipeline_summary.json",
        "resolved_pipeline_config.json",
        "ARTIFACT_MANIFEST.sha256",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def _resolved_config_for_artifact(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("__")}


def _read_capture_inventory(paths: Iterable[Path]) -> dict[str, Any]:
    record_count = 0
    ranks: set[int] = set()
    layers: set[int] = set()
    samples: dict[str, dict[int, set[int]]] = {}
    seen: set[tuple[str, int, int]] = set()
    duplicate_records: list[tuple[str, int, int]] = []
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid capture JSONL {path}:{line_no}: {exc}") from exc
            record_count += 1
            rank = int(payload.get("source_rank", -1))
            layer = int(payload.get("layer_id", -1))
            sample = str(payload.get("sample_id", ""))
            if rank < 0 or layer < 0 or not sample:
                raise RuntimeError(f"capture row lacks sample_id/source_rank/layer_id: {path}:{line_no}")
            ranks.add(rank)
            layers.add(layer)
            key = (sample, layer, rank)
            if key in seen:
                duplicate_records.append(key)
            seen.add(key)
            samples.setdefault(sample, {}).setdefault(layer, set()).add(rank)
    return {
        "record_count": record_count,
        "ranks": ranks,
        "layers": layers,
        "samples": samples,
        "duplicate_records": duplicate_records,
    }


def _jsonl_row_count(path: Path) -> int:
    count = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid capture JSONL {path}:{line_no}: {exc}") from exc
        count += 1
    return count


def inspect_capture_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = Path(config["output_dir"]) / "raw"
    routing_files = sorted(raw_dir.glob("*_source_expert_counts.jsonl")) if raw_dir.exists() else []
    manifest_files = sorted(raw_dir.glob("*_capture_manifest.json")) if raw_dir.exists() else []
    warning_files = sorted(raw_dir.glob("*_capture_warnings.jsonl")) if raw_dir.exists() else []
    inventory = _read_capture_inventory(routing_files) if routing_files else {
        "record_count": 0, "ranks": set(), "layers": set(), "samples": {}, "duplicate_records": []
    }
    warning_count = sum(_jsonl_row_count(path) for path in warning_files)
    accepted_source_ranks = config["capture"].get("accepted_source_ranks")
    expected_ranks = config["capture"].get("rank_to_node")
    expected_rank_count = None if expected_ranks is None else len(expected_ranks)
    expected_rank_set = (
        set(int(value) for value in accepted_source_ranks)
        if accepted_source_ranks is not None
        else set(range(expected_rank_count or 0))
    )
    expected_local_rank_count = len(expected_rank_set) if accepted_source_ranks is not None else expected_rank_count
    minimum_layers = int(config["capture"].get("minimum_consecutive_layers", 2))
    expected_count = config["capture"].get("expected_moe_layer_count")
    expected_ids = config["capture"].get("expected_layer_ids")
    require_all = bool(config["capture"].get("require_all_expected_layers", False))
    sample_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for sample_id, by_layer in sorted(inventory["samples"].items()):
        layer_ids = sorted(int(value) for value in by_layer)
        consecutive = bool(
            len(layer_ids) >= minimum_layers
            and all(layer_ids[index + 1] == layer_ids[index] + 1 for index in range(len(layer_ids) - 1))
        )
        incomplete_layers = [
            layer for layer, ranks in sorted(by_layer.items())
            if expected_local_rank_count is not None and set(ranks) != expected_rank_set
        ]
        expected_ok = True
        if require_all and expected_ids is not None:
            expected_ok = layer_ids == [int(value) for value in expected_ids]
        elif require_all and expected_count is not None:
            expected_ok = len(layer_ids) == int(expected_count)
        sample_status = consecutive and not incomplete_layers and expected_ok
        if not consecutive:
            failures.append(
                f"sample={sample_id} does not contain at least {minimum_layers} consecutive MoE layers: {layer_ids}"
            )
        if incomplete_layers:
            failures.append(
                f"sample={sample_id} lacks all EP source ranks for layers={incomplete_layers}"
            )
        if not expected_ok:
            failures.append(
                f"sample={sample_id} captured layers={layer_ids}, expected_ids={expected_ids}, expected_count={expected_count}"
            )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "layer_ids": layer_ids,
                "layer_count": len(layer_ids),
                "consecutive": consecutive,
                "incomplete_layers": incomplete_layers,
                "expected_layers_complete": expected_ok,
                "status": "PASS" if sample_status else "FAILED",
            }
        )
    status: dict[str, Any] = {
        "routing_file_count": len(routing_files),
        "routing_record_count": int(inventory["record_count"]),
        "manifest_file_count": len(manifest_files),
        "warning_record_count": warning_count,
        "source_ranks": sorted(inventory["ranks"]),
        "layer_ids": sorted(inventory["layers"]),
        "sample_count": len(sample_rows),
        "samples": sample_rows,
        "expected_source_rank_count": expected_local_rank_count,
        "expected_source_ranks": sorted(expected_rank_set),
        "expected_moe_layer_count": expected_count,
        "expected_layer_ids": expected_ids,
        "require_all_expected_layers": require_all,
        "duplicate_record_count": len(inventory["duplicate_records"]),
    }
    if int(inventory["record_count"]) <= 0:
        failures.append("no routing records were captured")
    if not sample_rows:
        failures.append("no complete capture sample was found")
    if expected_local_rank_count is not None and set(inventory["ranks"]) != expected_rank_set:
        failures.append(
            f"captured source ranks={sorted(inventory['ranks'])}, expected={sorted(expected_rank_set)}"
        )
    if inventory["duplicate_records"]:
        failures.append(
            f"duplicate sample/layer/source records detected: {inventory['duplicate_records'][:8]}"
        )
    if failures:
        status["status"] = "FAILED"
        status["failures"] = failures
    else:
        status["status"] = "PASS"
    return status


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_pipeline_artifacts(config: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(config["output_dir"])
    included: list[Path] = []
    excluded_roots = {"bundles", ".capture_bootstrap"}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        if relative.parts and relative.parts[0] in excluded_roots:
            continue
        if relative.name == "ARTIFACT_MANIFEST.sha256":
            continue
        included.append(path)
    manifest_path = output_dir / "ARTIFACT_MANIFEST.sha256"
    manifest_path.write_text(
        "".join(f"{_sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in included),
        encoding="utf-8",
    )
    included.append(manifest_path)
    bundle_dir = output_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in config["capture"]["capture_id"])
    bundle_path = bundle_dir / f"{safe_id}_trace_fixture_simulation.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in included:
            archive.write(path, path.relative_to(output_dir).as_posix())
    sha_path = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    sha_path.write_text(f"{_sha256_file(bundle_path)}  {bundle_path.name}\n", encoding="utf-8")
    return bundle_path, sha_path


def launch_collection(
    config: dict[str, Any],
    *,
    command_override: Iterable[str] | None = None,
) -> dict[str, Any]:
    command = list(command_override or config["launcher"].get("command", []))
    if not command:
        raise ValueError("no model launcher command; set launcher.command or pass a command after --")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(config["launcher"].get("clean_output_before_collect", True)):
        _clean_generated_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config["__config_path"])
    (output_dir / "resolved_pipeline_config.json").write_text(
        json.dumps(_resolved_config_for_artifact(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bootstrap = _bootstrap_dir(output_dir)
    env = os.environ.copy()
    env.update(config["launcher"].get("environment", {}))
    env["RS_SIM_CAPTURE_CONFIG"] = str(config_path)
    env["RS_SIM_CAPTURE_BACKEND"] = str(config["capture"]["backend"])
    env["RS_SIM_CAPTURE_DEFER_TO_DISTRIBUTED_WORKERS"] = "1"
    source_root = Path(__file__).resolve().parents[3]
    python_paths = [str(bootstrap), str(source_root)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    log_path = output_dir / "collect.log"
    started = time.time_ns()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            env=env,
            cwd=config.get("__config_dir") or None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=(int(config["launcher"].get("timeout_seconds", 0)) or None),
            check=False,
        )
    artifact_status = inspect_capture_artifacts(config) if process.returncode == 0 else {"status": "NOT_CHECKED"}
    result = {
        "schema_version": "RS_SIM_CAPTURE_LAUNCH_RESULT",
        "status": "PASS" if process.returncode == 0 else "FAILED",
        "returncode": process.returncode,
        "command": command,
        "log_path": str(log_path),
        "started_at_unix_ns": started,
        "finished_at_unix_ns": time.time_ns(),
        "capture_artifacts": artifact_status,
    }
    (output_dir / "collect_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if process.returncode != 0:
        raise RuntimeError(f"model collection command failed rc={process.returncode}; see {log_path}")
    if artifact_status.get("status") != "PASS":
        raise RuntimeError(
            "model command exited successfully but capture acceptance failed: "
            + "; ".join(artifact_status.get("failures", []))
            + f"; see {log_path}"
        )
    return result


def simulate_fixture(config: dict[str, Any], fixture_path: Path) -> Path:
    fixture = load_fixture(fixture_path)
    validate_fixture(fixture)
    simulation = config["simulation"]
    run_id = f"capture:{config['capture']['capture_id']}:{fixture.fixture_id}"
    runtime = build_current_p12_integration_runtime(
        fixture_input=fixture,
        run_id=run_id,
        staging_sensitivity=str(simulation["staging"]),
        release_mode=str(simulation["release"]),
        p0_p1_compute_end_barrier=bool(simulation.get("p0_p1_compute_end_barrier", PAPER_P0_P1_COMPUTE_END_BARRIER)),
        algorithm=str(simulation["algorithm"]),
        information_mode=str(simulation["information"]),
        overlap_mode=str(simulation["overlap"]),
        max_task_bytes=int(simulation["max_task_bytes"]),
    )
    try:
        timestamps = runtime.run_to_completion(max_timestamps=int(simulation["max_timestamps"]))
        runtime.assert_terminal()
        evidence = runtime.evidence()
        report = {
            "schema_version": "RS_SIM_CAPTURE_TO_FORMAL_RESULT",
            "status": "PASS",
            "capture_id": config["capture"]["capture_id"],
            "fixture_id": fixture.fixture_id,
            "fixture_path": str(fixture_path),
            "fixture_truth_digest": fixture.truth_digest(),
            "run_id": run_id,
            "axes": {
                "execution_line": "CURRENT_P12",
                "planning_window": "P12",
                "algorithm": simulation["algorithm"],
                "information": simulation["information"],
                "overlap": simulation["overlap"],
                "release": simulation["release"],
                "p0_p1_compute_end_barrier": bool(simulation.get("p0_p1_compute_end_barrier", PAPER_P0_P1_COMPUTE_END_BARRIER)),
                "staging": simulation["staging"],
            },
            "timestamps_processed": timestamps,
            "terminal": evidence["terminal_state"],
            "scheduler_runtime_metrics": evidence["scheduler_runtime_metrics"],
            "data_plane_runtime_metrics": evidence["data_plane_runtime_metrics"],
            "control_plane_runtime_metrics": evidence["control_plane_runtime_metrics"],
            "backend_phase_metrics": evidence["backend_phase_metrics"],
            "anchor_window_evidence": tuple(
                dataclasses.asdict(item) for item in runtime.current_p12_window_records()
            ),
            "formal_runtime_records": tuple(
                dataclasses.asdict(item) for item in runtime.formal_current_p12_records()
            ),
            "evidence_digest": stable_digest(evidence),
            "transport_transport": True,
            "performance_claim_allowed": False,
            "hardware_profile_calibrated": False,
        }
        result_dir = Path(config["output_dir"]) / "simulation"
        result_dir.mkdir(parents=True, exist_ok=True)
        path = result_dir / f"{fixture_path.stem}_formal_result.json"
        path.write_text(stable_json_dumps(report) + "\n", encoding="utf-8")
        return path
    finally:
        runtime.dispose()


def finalize_and_simulate(config: dict[str, Any]) -> dict[str, Any]:
    fixtures = build_fixtures_from_capture(config)
    results: list[Path] = []
    if bool(config["simulation"].get("enabled", True)):
        results = [simulate_fixture(config, path) for path in fixtures]
    summary = {
        "schema_version": "RS_SIM_CAPTURE_PIPELINE_SUMMARY",
        "status": "PASS",
        "capture_id": config["capture"]["capture_id"],
        "fixture_paths": [str(path) for path in fixtures],
        "simulation_result_paths": [str(path) for path in results],
        "fixture_count": len(fixtures),
        "simulation_count": len(results),
        "performance_claim_allowed": False,
    }
    path = Path(config["output_dir"]) / "pipeline_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle_path, bundle_sha_path = bundle_pipeline_artifacts(config)
    summary["artifact_bundle"] = str(bundle_path)
    summary["artifact_bundle_sha256_file"] = str(bundle_sha_path)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Rebuild once so the bundle contains the final summary that names the bundle.
    bundle_path, bundle_sha_path = bundle_pipeline_artifacts(config)
    summary["artifact_bundle"] = str(bundle_path)
    summary["artifact_bundle_sha256_file"] = str(bundle_sha_path)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
