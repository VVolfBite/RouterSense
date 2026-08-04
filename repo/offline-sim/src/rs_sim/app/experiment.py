from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

from rs_sim.runtime import (
    build_current_p12_integration_runtime,
    load_runtime_profile_bundle_json,
)
from rs_sim.contracts.digest import stable_digest, stable_json_dumps
from rs_sim.contracts.paper_defaults import (
    PAPER_ALIGNMENT_BYTES, PAPER_MAX_TASK_BYTES,
    PAPER_P0_P1_COMPUTE_END_BARRIER, PAPER_RELEASE_MODE,
    require_paper_execution_semantics,
    require_paper_treatment_release_semantics,
)
from rs_sim.trace.io.serialization import load_fixture
from rs_sim.scheduler.prediction.timing import load_rank_timing_profile
from rs_sim.trace.schema.validation import validate_fixture

from .config_io import (
    ConfigError, reject_unknown_fields, require_bool, require_mapping,
    require_nonempty, resolve_path,
)




# Keep terminated Popen objects alive until the CLI process exits via os._exit.
# Some numerical-runtime environments have exhibited blocking waitpid calls
# from Popen.poll()/__del__ even after the authoritative status/result files
# were committed.  Retaining the objects prevents destructor polling from
# stalling a long sweep.
_RETIRED_ISOLATED_PROCESSES: list[subprocess.Popen[Any]] = []


class RunProcessError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


def _freeze(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _normalize_treatment_name(value: Any, *, default: str) -> str:
    # YAML parses the bare token ``null`` as None.  Names are identifiers, not
    # null values, so use the deterministic default rather than the string
    # "None".
    if value is None:
        return str(default)
    text = str(value).strip()
    return text or str(default)




def _normalize_information(value: Any, scope: str) -> str:
    if scope == "PHASE_LOCAL":
        return "ZERO_P2"
    text = str(value).upper()
    aliases = {"PERFECT": "PERFECT_P2", "ZERO": "ZERO_P2", "FATE": "FATE_P2"}
    normalized = aliases.get(text, text)
    if normalized not in {"FATE_P2", "PERFECT_P2", "ZERO_P2"}:
        raise ConfigError(
            "joint information must be FATE_P2, PERFECT_P2, or ZERO_P2; "
            "last-value prediction is not a formal route"
        )
    return normalized


def _normalize_overlap(value: Any) -> str:
    return str(value).upper()


def normalize_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    reject_unknown_fields(
        config,
        {"version", "name", "trace", "traces", "simulation", "experiments",
         "repetitions", "output", "execution", "comparison", "oracle",
         "__config_path", "__config_dir"},
        "experiment config",
    )
    if int(config.get("version", 1)) != 1:
        raise ConfigError("experiment config version must be 1")
    name = require_nonempty(config.get("name", "experiment"), "name")
    traces_value = config.get("traces", config.get("trace"))
    if isinstance(traces_value, (str, Path)):
        traces_value = [traces_value]
    if not isinstance(traces_value, list) or not traces_value:
        raise ConfigError("trace/traces must specify at least one path")
    trace_paths = [resolve_path(str(item), config=config) for item in traces_value]
    simulation = require_mapping(config.get("simulation", {}), "simulation")
    experiments = require_mapping(config.get("experiments"), "experiments")
    repetitions = require_mapping(config.get("repetitions", {}), "repetitions")
    output_cfg = require_mapping(config.get("output"), "output")
    execution_cfg = require_mapping(config.get("execution", {}), "execution")
    comparison_cfg = require_mapping(config.get("comparison", {}), "comparison")
    oracle_cfg = require_mapping(config.get("oracle", {}), "oracle")
    reject_unknown_fields(simulation, {
        "release_mode", "release", "p0_p1_compute_end_barrier", "staging",
        "max_task_bytes", "max_window_prefix_tasks", "alignment_bytes",
        "max_timestamps", "runtime_profile", "rank_timing_profile",
    }, "simulation")
    reject_unknown_fields(experiments, {"treatments"}, "experiments")
    reject_unknown_fields(repetitions, {"warmup", "measure", "seed"}, "repetitions")
    reject_unknown_fields(output_cfg, {
        "directory", "overwrite", "save_raw_events", "save_task_timeline",
        "save_plans", "complete_csv_filename", "raw_only",
    }, "output")
    reject_unknown_fields(execution_cfg, {
        "mode", "per_run_timeout_seconds", "kill_grace_seconds", "fail_fast",
    }, "execution")
    reject_unknown_fields(comparison_cfg, {
        "claim_mode", "baseline", "baselines_by_overlap", "reference_baselines",
        "primary_metric", "target_improvement_percent", "tie_tolerance_percent",
        "minimum_paired_samples",
    }, "comparison")
    reject_unknown_fields(oracle_cfg, {
        "time_limit_ms_per_window", "relative_gap", "require_all_certified",
    }, "oracle")
    output_dir = resolve_path(require_nonempty(output_cfg.get("directory"), "output.directory"), config=config)
    execution_mode = str(execution_cfg.get("mode", "SUBPROCESS_ISOLATED")).upper()
    if execution_mode not in {"SUBPROCESS_ISOLATED", "INPROCESS_DEBUG", "INPROCESS_DURABLE"}:
        raise ConfigError(
            "execution.mode must be SUBPROCESS_ISOLATED, INPROCESS_DEBUG, or INPROCESS_DURABLE"
        )
    per_run_timeout = int(execution_cfg.get("per_run_timeout_seconds", 1200))
    kill_grace = int(execution_cfg.get("kill_grace_seconds", 10))
    if per_run_timeout <= 0 or kill_grace <= 0:
        raise ConfigError("execution timeouts must be positive")
    target_improvement = float(comparison_cfg.get("target_improvement_percent", 30.0))
    if not math.isfinite(target_improvement):
        raise ConfigError("comparison.target_improvement_percent must be finite")
    raw_baselines_by_overlap = comparison_cfg.get("baselines_by_overlap", {})
    if raw_baselines_by_overlap is None:
        raw_baselines_by_overlap = {}
    if not isinstance(raw_baselines_by_overlap, dict):
        raise ConfigError("comparison.baselines_by_overlap must be a mapping")
    baselines_by_overlap = {
        _normalize_overlap(mode): str(treatment).strip()
        for mode, treatment in raw_baselines_by_overlap.items()
        if str(treatment).strip()
    }
    raw_references = comparison_cfg.get("reference_baselines", [])
    if isinstance(raw_references, str):
        raw_references = [raw_references]
    if not isinstance(raw_references, list):
        raise ConfigError("comparison.reference_baselines must be a list")
    reference_baselines = tuple(
        str(value).strip() for value in raw_references if str(value).strip()
    )
    oracle_time_limit_ms = int(oracle_cfg.get("time_limit_ms_per_window", 30_000))
    oracle_relative_gap = float(oracle_cfg.get("relative_gap", 0.0))
    if oracle_time_limit_ms <= 0:
        raise ConfigError("oracle.time_limit_ms_per_window must be positive")
    if not math.isfinite(oracle_relative_gap) or not 0.0 <= oracle_relative_gap < 1.0:
        raise ConfigError("oracle.relative_gap must be finite and in [0, 1)")
    release_mode = str(
        simulation.get("release_mode", simulation.get("release", PAPER_RELEASE_MODE))
    ).upper()
    if release_mode not in {"RANK_LOCAL", "PHASE_BARRIER"}:
        raise ConfigError("simulation.release_mode must be RANK_LOCAL or PHASE_BARRIER")
    max_task_bytes = int(simulation.get("max_task_bytes", PAPER_MAX_TASK_BYTES))
    alignment_bytes = int(simulation.get("alignment_bytes", PAPER_ALIGNMENT_BYTES))
    if max_task_bytes <= 0 or alignment_bytes <= 0:
        raise ConfigError("simulation task and alignment sizes must be positive")

    timing_profile_value = simulation.get("rank_timing_profile")
    timing_profile_path = (
        None
        if timing_profile_value in (None, "")
        else resolve_path(str(timing_profile_value), config=config)
    )
    return {
        "name": name,
        "trace_paths": trace_paths,
        "simulation": {
            "release_mode": release_mode,
            "p0_p1_compute_end_barrier": require_bool(simulation.get("p0_p1_compute_end_barrier"), "simulation.p0_p1_compute_end_barrier", default=PAPER_P0_P1_COMPUTE_END_BARRIER),
            "staging": str(simulation.get("staging", "1.0X")).upper(),
            "max_task_bytes": max_task_bytes,
            "max_window_prefix_tasks": int(simulation.get("max_window_prefix_tasks", 256)),
            "alignment_bytes": alignment_bytes,
            "max_timestamps": int(simulation.get("max_timestamps", 100000)),
            "runtime_profile": (
                None
                if simulation.get("runtime_profile") is None
                else resolve_path(str(simulation["runtime_profile"]), config=config)
            ),
            "rank_timing_profile": timing_profile_path,
        },
        "experiments": experiments,
        "repetitions": {
            "warmup": int(repetitions.get("warmup", 0)),
            "measure": int(repetitions.get("measure", 1)),
            "seed": int(repetitions.get("seed", 1234)),
        },
        "execution": {
            "mode": execution_mode,
            "per_run_timeout_seconds": per_run_timeout,
            "kill_grace_seconds": kill_grace,
            "fail_fast": require_bool(execution_cfg.get("fail_fast"), "execution.fail_fast", default=False),
        },
        "oracle": {
            "time_limit_ms_per_window": oracle_time_limit_ms,
            "relative_gap": oracle_relative_gap,
            "require_all_certified": require_bool(oracle_cfg.get("require_all_certified"), "oracle.require_all_certified", default=True),
        },
        "comparison": {
            "claim_mode": str(comparison_cfg.get("claim_mode", "DIAGNOSTIC")).strip().upper(),
            "baseline": (
                "" if comparison_cfg.get("baseline") is None
                else str(comparison_cfg.get("baseline", "")).strip()
            ),
            "baselines_by_overlap": baselines_by_overlap,
            "reference_baselines": reference_baselines,
            "primary_metric": str(comparison_cfg.get("primary_metric", "window_makespan_ns_sum")),
            "target_improvement_percent": target_improvement,
            "tie_tolerance_percent": float(comparison_cfg.get("tie_tolerance_percent", 0.05)),
            "minimum_paired_samples": int(comparison_cfg.get("minimum_paired_samples", 3)),
        },
        "output_dir": output_dir,
        "overwrite": require_bool(output_cfg.get("overwrite"), "output.overwrite", default=False),
        "save_raw_events": require_bool(output_cfg.get("save_raw_events"), "output.save_raw_events", default=False),
        "save_task_timeline": require_bool(output_cfg.get("save_task_timeline"), "output.save_task_timeline", default=False),
        "save_plans": require_bool(output_cfg.get("save_plans"), "output.save_plans", default=False),
        "complete_csv_filename": str(output_cfg.get("complete_csv_filename", "all_results.csv")).strip() or "all_results.csv",
        "raw_only": require_bool(output_cfg.get("raw_only"), "output.raw_only", default=False),
        "__source": config,
    }


def _audit_treatment(item: dict[str, str]) -> dict[str, str]:
    labels = {
        "null": "Null", "fifo": "FIFO", "greedy": "Largest-First Greedy",
        "birkhoff": "BvN-style", "islip": "iSLIP-style-4",
        "residual_mwm": "Residual-MWM", "fast": "FAST-style",
        "aurora": "Aurora-style", "rscf": "RSCF", "oracle": "Oracle",
    }
    enriched = dict(item)
    enriched.update({
        "paper_label": labels[item["core"]],
        "comparison_class": "PROJECT_NATIVE" if item["core"] in {"rscf", "oracle"} else "REFERENCE_BASELINE",
        "paper_claim_allowed": "true",
        "main_table_allowed": "true",
        "style_required": "true" if item["core"] in {"birkhoff", "islip", "fast", "aurora"} else "false",
    })
    return enriched


def _validate_paper_claim_treatments(treatments: tuple[dict[str, str], ...]) -> None:
    names: set[str] = set()
    for item in treatments:
        if item["name"] in names:
            raise ConfigError(f"duplicate treatment name {item['name']!r}")
        names.add(item["name"])


def _expand_treatments(
    experiments: dict[str, Any], *, default_release_mode: str
) -> tuple[dict[str, str], ...]:
    from rs_sim.scheduler.decorators.composition import parse_algorithm_expression

    forbidden_matrix_keys = {"baselines", "releasefrontier", "rscf"}
    present = sorted(key for key in forbidden_matrix_keys if key in experiments)
    if present:
        raise ConfigError(
            "legacy experiment matrix keys are not supported: " + ", ".join(present)
        )
    explicit = experiments.get("treatments", [])
    if not isinstance(explicit, list) or not explicit:
        raise ConfigError("experiments.treatments must be a non-empty list")

    result: list[dict[str, str]] = []
    legacy_fields = {"core", "policy", "scope", "planning", "safe_selector"}
    for row in explicit:
        if not isinstance(row, dict):
            raise ConfigError("experiments.treatments entries must be objects")
        reject_unknown_fields(row, {
            "name", "algorithm", "information", "overlap", "experiment_role",
            "release_mode", "allow_matched_joint_diagnostic", *legacy_fields,
        }, "experiments.treatments[]")
        used_legacy = sorted(legacy_fields.intersection(row))
        if used_legacy:
            raise ConfigError(
                "treatment must use one algorithm expression; remove legacy fields: "
                + ", ".join(used_legacy)
            )
        expression = require_nonempty(row.get("algorithm"), "treatment.algorithm")
        try:
            algorithm = parse_algorithm_expression(expression)
        except ValueError as exc:
            raise ConfigError(f"invalid treatment.algorithm {expression!r}: {exc}") from exc
        scope = algorithm.scope.value
        release_mode = str(
            row.get("release_mode", default_release_mode)
        ).upper()
        if release_mode not in {"RANK_LOCAL", "PHASE_BARRIER"}:
            raise ConfigError(
                "treatment.release_mode must be RANK_LOCAL or PHASE_BARRIER"
            )
        information = _normalize_information(row.get("information", "FATE_P2"), scope)
        if scope == "PHASE_LOCAL" and information != "ZERO_P2":
            raise ConfigError("Local treatments must use ZERO_P2")
        if algorithm.core_id == "oracle" and scope == "WINDOW_JOINT" and information != "PERFECT_P2":
            raise ConfigError("Joint Oracle requires PERFECT_P2")
        item = {
            "name": _normalize_treatment_name(
                row.get("name"), default=algorithm.expression
            ),
            "algorithm": algorithm.expression,
            "core": algorithm.core_id,
            "policy": algorithm.core_id,
            "scope": scope,
            "planning": algorithm.planning.value,
            "information": information,
            "release_mode": release_mode,
            "overlap": _normalize_overlap(row.get("overlap", "OVERLAP")),
            "safe_selector": algorithm.safe,
            "experiment_role": str(
                row.get("experiment_role", "EXPLICIT_DIAGNOSTIC_ABLATION")
            ).upper(),
            "allow_matched_joint_diagnostic": require_bool(
                row.get("allow_matched_joint_diagnostic"),
                "treatment.allow_matched_joint_diagnostic",
                default=False,
            ),
        }
        if (
            scope == "WINDOW_JOINT"
            and algorithm.core_id not in {"rscf", "oracle"}
            and not item["allow_matched_joint_diagnostic"]
        ):
            raise ConfigError(
                f"{item['name']!r} wraps external core {algorithm.core_id!r} in Joint. "
                "External baselines must remain Local unless an explicit diagnostic sets "
                "allow_matched_joint_diagnostic: true."
            )
        result.append(_audit_treatment(item))

    unique: dict[str, dict[str, str]] = {}
    for item in result:
        key = stable_digest({
            field: item[field]
            for field in ("algorithm", "information", "release_mode", "overlap")
        })
        if key in unique:
            raise ConfigError(
                f"duplicate treatment semantics: {item['name']!r} and {unique[key]['name']!r}"
            )
        unique[key] = item
    return tuple(unique.values())


def _extract_trace_bundle(path: Path, temporary_roots: list[tempfile.TemporaryDirectory]) -> Path:
    temp = tempfile.TemporaryDirectory(prefix="rs-sim-trace-")
    temporary_roots.append(temp)
    target = Path(temp.name)
    lowered = path.name.lower()
    if lowered.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                candidate = (target / member.filename).resolve()
                if target.resolve() not in candidate.parents and candidate != target.resolve():
                    raise ConfigError(f"trace archive contains unsafe path: {member.filename}")
            archive.extractall(target)
        return target
    if lowered.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive.getmembers():
                candidate = (target / member.name).resolve()
                if target.resolve() not in candidate.parents and candidate != target.resolve():
                    raise ConfigError(f"trace archive contains unsafe path: {member.name}")
                if member.issym() or member.islnk():
                    raise ConfigError(f"trace archive contains link entry: {member.name}")
            archive.extractall(target)
        return target
    raise ConfigError(f"unsupported trace archive: {path}")


def _manifest_fixture_paths(path: Path, payload: dict[str, Any]) -> tuple[Path, ...]:
    """Resolve fixture paths from measured, projected, or matrix manifests."""
    raw_entries: Any = payload.get("fixtures")
    if raw_entries is None:
        raw_entries = payload.get("fixture_paths")
    if raw_entries is None:
        return ()
    if not isinstance(raw_entries, list):
        raise ConfigError(f"trace manifest fixture list must be a list: {path}")
    base = path.parent
    resolved: list[Path] = []
    for entry in raw_entries:
        if isinstance(entry, str):
            relative = entry
        elif isinstance(entry, dict):
            relative = entry.get("path", entry.get("fixture_path"))
        else:
            relative = None
        if not str(relative or "").strip():
            raise ConfigError(f"trace manifest contains an invalid fixture entry: {path}")
        candidate = (base / str(relative)).resolve()
        if not candidate.is_file():
            raise ConfigError(f"trace manifest fixture does not exist: {candidate}")
        resolved.append(candidate)
    return tuple(sorted(set(resolved), key=str))


def _recursive_repository_fixtures(
    path: Path,
    temporary_roots: list[tempfile.TemporaryDirectory],
) -> tuple[Path, ...]:
    """Resolve a consolidated trace repository without scanning unrelated JSON."""
    manifests = tuple(sorted(path.rglob("trace_manifest.json"), key=str))
    fixtures: list[Path] = []
    for manifest in manifests:
        fixtures.extend(_fixtures_from_path(manifest, temporary_roots))
    if fixtures:
        return tuple(sorted(set(fixtures), key=str))

    # Compatibility fallback for repositories that preserve fixtures but omit
    # per-configuration manifests.  Restrict discovery to directories named
    # ``fixtures`` so reports, raw captures, and catalog JSON cannot be loaded
    # as workload fixtures by accident.
    candidates = tuple(sorted(path.rglob("fixtures/*.json"), key=str))
    if candidates:
        return candidates
    return ()


def _trace_identity_fields(fixture_path: Path, fixture: Any) -> dict[str, Any]:
    """Preserve dataset provenance and consolidated-repository coordinates.

    Supported repository layouts:

    * legacy: ``measured/EP4/olmoe/s128/fixtures/...``
    * model-first: ``OLMoE-1B-7B-0924/EP4/SEQ128/fixtures/...``

    The model-first layout reads ``REORGANIZED_LOCATION.json`` (or the local
    manifest schema as a fallback) to preserve measured/projected provenance.
    """
    resolved_path = fixture_path.resolve()
    parts = resolved_path.parts
    trace_mode: str | None = None
    trace_ep: int | None = None
    trace_model: str | None = None
    trace_sequence_length: int | None = None

    # Legacy consolidated repository layout.
    for index, part in enumerate(parts):
        lowered = part.lower()
        if lowered not in {"measured", "projected"} or index + 3 >= len(parts):
            continue
        ep_part = parts[index + 1]
        sequence_part = parts[index + 3]
        if not ep_part.upper().startswith("EP") or not ep_part[2:].isdigit():
            continue
        sequence_lower = sequence_part.lower()
        if not sequence_lower.startswith("s") or not sequence_lower[1:].isdigit():
            continue
        trace_mode = lowered
        trace_ep = int(ep_part[2:])
        trace_model = parts[index + 2]
        trace_sequence_length = int(sequence_lower[1:])
        break

    # User-facing model-first repository layout.
    if trace_ep is None:
        for sequence_dir in resolved_path.parents:
            sequence_name = sequence_dir.name
            sequence_upper = sequence_name.upper()
            if sequence_upper.startswith("SEQ") and sequence_upper[3:].isdigit():
                sequence_value = int(sequence_upper[3:])
            elif sequence_name.lower().startswith("s") and sequence_name[1:].isdigit():
                sequence_value = int(sequence_name[1:])
            else:
                continue
            ep_dir = sequence_dir.parent
            ep_name = ep_dir.name.upper()
            if not ep_name.startswith("EP") or not ep_name[2:].isdigit():
                continue
            model_dir = ep_dir.parent
            trace_ep = int(ep_name[2:])
            trace_model = model_dir.name
            trace_sequence_length = sequence_value

            marker = sequence_dir / "REORGANIZED_LOCATION.json"
            if marker.is_file():
                try:
                    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    marker_payload = {}
                original_path = str(marker_payload.get("original_path", "")).lower()
                status = str(marker_payload.get("status", "")).lower()
                origin = str(marker_payload.get("origin", "")).lower()
                if original_path.startswith("measured/") or status.startswith("measured") or "measured" in origin:
                    trace_mode = "measured"
                elif original_path.startswith("projected/") or status.startswith("projected") or "project" in origin:
                    trace_mode = "projected"

            if trace_mode is None:
                manifest = sequence_dir / "trace_manifest.json"
                if manifest.is_file():
                    try:
                        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        manifest_payload = {}
                    schema = str(manifest_payload.get("schema_version", "")).upper()
                    trace_mode = "projected" if "PROJECTED" in schema else "measured"
            break

    provenance = fixture.provenance
    return {
        "trace_mode": trace_mode,
        "trace_ep": trace_ep,
        "trace_model": trace_model,
        "trace_sequence_length": trace_sequence_length,
        "trace_fixture_name": fixture_path.name,
        "trace_dataset_id": provenance.dataset_id,
        "trace_split": provenance.split,
        "trace_source_digest": provenance.source_digest,
        "trace_transform_digest": provenance.transform_digest,
        "trace_capture_id": provenance.capture_id,
        "trace_collector_version": provenance.collector_version,
        "trace_source_kind": provenance.source_kind,
        "trace_notes": provenance.notes,
        "trace_provenance_digest": provenance.identity_digest(),
    }


def _fixtures_from_path(path: Path, temporary_roots: list[tempfile.TemporaryDirectory]) -> tuple[Path, ...]:
    if not path.exists():
        raise ConfigError(f"trace path does not exist: {path}")
    lowered_name = path.name.lower()
    if lowered_name.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        return _fixtures_from_path(_extract_trace_bundle(path, temporary_roots), temporary_roots)
    if path.is_dir():
        manifest = path / "trace_manifest.json"
        if manifest.is_file():
            return _fixtures_from_path(manifest, temporary_roots)
        repository_fixtures = _recursive_repository_fixtures(path, temporary_roots)
        if repository_fixtures:
            return repository_fixtures
        candidates = tuple(sorted((path / "fixtures").glob("*.json"))) or tuple(sorted(path.glob("*.json")))
        if not candidates:
            raise ConfigError(f"no fixture JSON found in trace directory {path}")
        return candidates
    payload = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "TRACE_CATALOG.json":
        repository_fixtures = _recursive_repository_fixtures(path.parent, temporary_roots)
        if not repository_fixtures:
            raise ConfigError(f"trace catalog resolves no fixtures: {path}")
        return repository_fixtures
    manifest_fixtures = _manifest_fixture_paths(path, payload)
    if manifest_fixtures:
        return manifest_fixtures
    return (path,)



def _validate_fate_joint_runtime_contract(
    treatment: dict[str, Any], windows: Iterable[Any]
) -> None:
    """Fail closed when a FATE-Joint run silently loses its P2 advisory.

    The main ReleaseFrontier treatment is required to generate, validate and
    consume a non-empty FATE P2 template whenever the realized P2 phase
    contains remote tasks. A poor prediction may legitimately require repair;
    an empty or silently downgraded prediction may not.
    """

    if str(treatment.get("scope", "")).upper() != "WINDOW_JOINT":
        return
    if str(treatment.get("information", "")).upper() != "FATE_P2":
        return
    for item in windows:
        details = {
            "treatment": str(treatment.get("name", treatment.get("policy", "unknown"))),
            "anchor_layer_id": int(item.anchor_layer_id),
            "information_mode": str(item.information_mode),
            "predicted_p2_slot_count": int(item.predicted_p2_slot_count),
            "bound_exact_p2_task_count": int(item.bound_exact_p2_task_count),
            "prediction_generated": bool(item.prediction_generated),
            "prediction_nonempty": bool(item.prediction_nonempty),
            "prediction_validated": bool(item.prediction_validated),
            "prediction_consumed": bool(item.prediction_consumed),
            "prediction_fallback": bool(item.prediction_fallback),
            "prediction_fallback_reason": item.prediction_fallback_reason,
        }
        if str(item.information_mode).upper() != "FATE_P2":
            raise RunProcessError(
                "FATE-Joint treatment changed information mode at runtime",
                details=details,
            )
        if not bool(item.prediction_generated):
            raise RunProcessError(
                "FATE-Joint treatment did not generate a P2 prediction",
                details=details,
            )
        if not bool(item.prediction_validated):
            raise RunProcessError(
                "FATE-Joint treatment did not validate its P2 prediction",
                details=details,
            )
        if not bool(item.prediction_consumed):
            raise RunProcessError(
                "FATE-Joint treatment did not consume its P2 prediction",
                details=details,
            )
        if bool(item.prediction_fallback):
            raise RunProcessError(
                "FATE-Joint treatment silently fell back from FATE P2",
                details=details,
            )
        if int(item.bound_exact_p2_task_count) > 0 and (
            int(item.predicted_p2_slot_count) <= 0 or not bool(item.prediction_nonempty)
        ):
            raise RunProcessError(
                "FATE-Joint treatment produced an empty P2 template despite remote P2 tasks",
                details=details,
            )



def _validate_fate_joint_identity(rows: Iterable[dict[str, Any]]) -> None:
    """Require every matched FATE-Joint treatment to receive the same advisory.

    This check is mostly relevant to explicitly opted-in diagnostic wrappers.
    The default matrix contains only RSCF-Joint, but keeping the identity gate
    prevents future experiments from comparing Joint algorithms under different
    P2 information.
    """

    signatures: dict[tuple[str, int], tuple[tuple[int, str], ...]] = {}
    owners: dict[tuple[str, int], str] = {}
    for row in rows:
        treatment = dict(row.get("treatment", {}))
        if str(treatment.get("scope", "")).upper() != "WINDOW_JOINT":
            continue
        if str(treatment.get("information", "")).upper() != "FATE_P2":
            continue
        key = (str(row.get("fixture_truth_digest", "")), int(row.get("repeat_index", 0)))
        signature = tuple(
            (int(item.get("anchor_layer_id", -1)), str(item.get("prediction_digest", "")))
            for item in row.get("per_window_metrics", ())
        )
        if key not in signatures:
            signatures[key] = signature
            owners[key] = str(treatment.get("name", treatment.get("policy", "unknown")))
            continue
        if signatures[key] != signature:
            raise RunProcessError(
                "FATE-Joint treatments received different P2 advisory digests",
                details={
                    "fixture_truth_digest": key[0],
                    "repeat_index": key[1],
                    "reference_treatment": owners[key],
                    "reference_signature": signatures[key],
                    "mismatched_treatment": str(
                        treatment.get("name", treatment.get("policy", "unknown"))
                    ),
                    "mismatched_signature": signature,
                },
            )


def _run_one(
    *,
    fixture_path: Path,
    fixture_index: int,
    treatment: dict[str, str],
    treatment_index: int,
    repeat_index: int,
    warmup: bool,
    config: dict[str, Any],
    run_dir: Path,
    dispose_runtime: bool = True,
    trusted_fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wall_started_ns = time.monotonic_ns()
    fixture_validation_started_ns = time.monotonic_ns()
    validation_mode = "FULL_WORKER_VALIDATION"
    if trusted_fixture is None:
        fixture = load_fixture(fixture_path)
        validate_fixture(fixture)
        fixture_truth_digest = fixture.truth_digest()
    else:
        expected_size = int(trusted_fixture["size_bytes"])
        expected_mtime = int(trusted_fixture["mtime_ns"])
        expected_digest = str(trusted_fixture["truth_digest"])
        current_stat = fixture_path.stat()
        if int(current_stat.st_size) != expected_size or int(current_stat.st_mtime_ns) != expected_mtime:
            raise RunProcessError(
                "fixture changed after parent validation",
                details={
                    "fixture_path": str(fixture_path),
                    "expected_size_bytes": expected_size,
                    "actual_size_bytes": int(current_stat.st_size),
                    "expected_mtime_ns": expected_mtime,
                    "actual_mtime_ns": int(current_stat.st_mtime_ns),
                },
            )
        fixture = load_fixture(fixture_path, verify_declared_digest=False)
        fixture_truth_digest = fixture.truth_digest()
        if fixture_truth_digest != expected_digest:
            raise RunProcessError(
                "fixture truth digest changed after parent validation",
                details={
                    "fixture_path": str(fixture_path),
                    "expected_fixture_truth_digest": expected_digest,
                    "actual_fixture_truth_digest": fixture_truth_digest,
                },
            )
        validation_mode = "PARENT_VALIDATED_TRUSTED_WORKER"
    fixture_validation_elapsed_ns = time.monotonic_ns() - fixture_validation_started_ns
    paired_instance_id = stable_digest(
        {
            "schema_version": "RS_SIM_EXPERIMENT_PAIRED_INSTANCE",
            "fixture_truth_digest": fixture_truth_digest,
            "repeat_index": repeat_index,
            "max_task_bytes": config["simulation"]["max_task_bytes"],
            "max_window_prefix_tasks": config["simulation"]["max_window_prefix_tasks"],
            "alignment_bytes": config["simulation"]["alignment_bytes"],
        }
    )
    run_id = f"{config['name']}:{fixture_index:03d}:{treatment_index:03d}:{repeat_index:03d}:{'warmup' if warmup else 'measure'}"
    simulation_run_id = f"paired:{paired_instance_id}"
    timing_profile_path = config["simulation"].get("rank_timing_profile")
    rank_timing_profile = (
        None
        if timing_profile_path is None
        else load_rank_timing_profile(Path(timing_profile_path))
    )
    if rank_timing_profile is not None:
        rank_timing_profile.assert_compatible(fixture)
    runtime_profile_path = config["simulation"].get("runtime_profile")
    runtime_profile = (
        None
        if runtime_profile_path is None
        else load_runtime_profile_bundle_json(Path(runtime_profile_path))
    )
    runtime = build_current_p12_integration_runtime(
        fixture_input=fixture,
        run_id=simulation_run_id,
        paired_instance_id=paired_instance_id,
        staging_sensitivity=config["simulation"]["staging"],
        release_mode=treatment["release_mode"],
        p0_p1_compute_end_barrier=config["simulation"]["p0_p1_compute_end_barrier"],
        algorithm=treatment["algorithm"],
        information_mode=treatment["information"],
        overlap_mode=treatment["overlap"],
        max_task_bytes=config["simulation"]["max_task_bytes"],
        max_window_prefix_tasks=config["simulation"]["max_window_prefix_tasks"],
        alignment_bytes=config["simulation"]["alignment_bytes"],
        runtime_profile=runtime_profile,
        rank_timing_profile=rank_timing_profile,
        oracle_time_limit_ms=int(config["oracle"]["time_limit_ms_per_window"]),
        oracle_relative_gap=float(config["oracle"]["relative_gap"]),
        oracle_require_certified=bool(config["oracle"]["require_all_certified"]),
    )
    try:
        timestamps = runtime.run_to_completion(max_timestamps=config["simulation"]["max_timestamps"])
        runtime.assert_terminal()
        windows = runtime.current_p12_window_records()
        _validate_fate_joint_runtime_contract(treatment, windows)
        formal = runtime.formal_current_p12_records()
        evidence = runtime.evidence()
        timeline = runtime.data_plane.formal_runtime_metrics()["statistics"]["transfer_timeline"]
        window_makespans = [int(item.window_makespan_ns) for item in windows]
        network_transfer_spans = [int(item.network_transfer_span_ns) for item in windows]
        compute_excluded_communication_makespans = [
            int(item.compute_excluded_communication_makespan_ns) for item in windows
        ]
        network_active_unions = [int(item.network_active_union_ns) for item in windows]
        rank_communication_exposed_max_values = [
            int(item.rank_communication_exposed_ns_max) for item in windows
        ]
        rank_communication_exposed_all_values = [
            int(value)
            for item in windows
            for value in item.rank_communication_exposed_ns_by_rank
        ]
        rank_communication_exposed_window_spreads = [
            max((int(value) for value in item.rank_communication_exposed_ns_by_rank), default=0)
            - min((int(value) for value in item.rank_communication_exposed_ns_by_rank), default=0)
            for item in windows
        ]
        p0_p1_local_completion_spreads = []
        for item in windows:
            local_completion = [
                int(value) for value in item.p0_p1_local_complete_times_ns if value is not None
            ]
            p0_p1_local_completion_spreads.append(
                max(local_completion, default=0) - min(local_completion, default=0)
            )
        prediction_relative_errors = [
            int(item.prediction_relative_absolute_error_ppm or 0) for item in windows
        ]
        prediction_overlaps = [int(item.prediction_matrix_overlap_ppm or 0) for item in windows]
        prediction_top_destination = [
            int(item.prediction_top_destination_accuracy_ppm or 0) for item in windows
        ]
        p2_first_release_offsets: list[int] = []
        p2_last_release_offsets: list[int] = []
        p2_release_spreads: list[int] = []
        all_p2_release_offsets: list[int] = []
        per_window_metrics: list[dict[str, Any]] = []
        for item in windows:
            p2_digest = stable_digest(item.p2_dispatch_phase_key)
            offsets = sorted(
                int(at_ns) - int(item.window_start_ns)
                for phase_digest, _rank, at_ns in item.rank_release_times_ns
                if str(phase_digest) == str(p2_digest)
            )
            if offsets:
                p2_first_release_offsets.append(offsets[0])
                p2_last_release_offsets.append(offsets[-1])
                p2_release_spreads.append(offsets[-1] - offsets[0])
                all_p2_release_offsets.extend(offsets)
            per_window_metrics.append(
                {
                    "anchor_layer_id": int(item.anchor_layer_id),
                    "truth_digest": str(item.truth_digest),
                    "task_catalogue_digest": str(item.task_catalogue_digest),
                    "task_boundary_digest": str(item.task_boundary_digest),
                    "window_makespan_ns": int(item.window_makespan_ns),
                    "network_transfer_span_ns": int(item.network_transfer_span_ns),
                    "compute_excluded_communication_makespan_ns": int(item.compute_excluded_communication_makespan_ns),
                    "network_active_union_ns": int(item.network_active_union_ns),
                    "rank_communication_exposed_p1_ns_by_rank": item.rank_communication_exposed_p1_ns_by_rank,
                    "rank_communication_exposed_p2_ns_by_rank": item.rank_communication_exposed_p2_ns_by_rank,
                    "rank_communication_exposed_ns_by_rank": item.rank_communication_exposed_ns_by_rank,
                    "rank_communication_exposed_ns_sum": int(item.rank_communication_exposed_ns_sum),
                    "rank_communication_exposed_ns_mean": int(item.rank_communication_exposed_ns_mean),
                    "rank_communication_exposed_ns_max": int(item.rank_communication_exposed_ns_max),
                    "rank_communication_exposed_ns_p95": int(item.rank_communication_exposed_ns_p95),
                    "rank_communication_exposed_ns_p99": int(item.rank_communication_exposed_ns_p99),
                    # Stable user-facing aliases for the causal per-rank
                    # communication exposure.  These exclude the shared compute
                    # spans but are not the full frozen-plan Runtime dual run.
                    "communication_stall_ns_by_rank": item.rank_communication_exposed_ns_by_rank,
                    "mean_communication_stall_ns": int(item.rank_communication_exposed_ns_mean),
                    "p95_communication_stall_ns": int(item.rank_communication_exposed_ns_p95),
                    "max_communication_stall_ns": int(item.rank_communication_exposed_ns_max),
                    "rank_communication_exposed_ns_min": min((int(value) for value in item.rank_communication_exposed_ns_by_rank), default=0),
                    "rank_communication_exposed_ns_spread": max((int(value) for value in item.rank_communication_exposed_ns_by_rank), default=0) - min((int(value) for value in item.rank_communication_exposed_ns_by_rank), default=0),
                    "rank_communication_critical_rank": item.rank_communication_critical_rank,
                    "p2_first_rank_release_offset_ns": offsets[0] if offsets else None,
                    "p2_last_rank_release_offset_ns": offsets[-1] if offsets else None,
                    "p2_rank_release_spread_ns": (offsets[-1] - offsets[0]) if offsets else None,
                    "p0_p1_local_completion_spread_ns": (
                        max((int(value) for value in item.p0_p1_local_complete_times_ns if value is not None), default=0)
                        - min((int(value) for value in item.p0_p1_local_complete_times_ns if value is not None), default=0)
                    ),
                    "prediction_digest": str(item.prediction_digest),
                    "information_mode": str(item.information_mode),
                    "prediction_relative_absolute_error_ppm": item.prediction_relative_absolute_error_ppm,
                    "prediction_matrix_overlap_ppm": item.prediction_matrix_overlap_ppm,
                    "prediction_top_destination_accuracy_ppm": item.prediction_top_destination_accuracy_ppm,
                    "prediction_exposed_ns": int(item.prediction_exposed_ns),
                    "control_exposed_ns": int(item.control_exposed_ns),
                    "binding_exposed_ns": int(item.binding_exposed_ns),
                    "algorithm_core_run_count": int(item.algorithm_core_run_count),
                    "incremental_bind_job_count": int(item.incremental_bind_job_count),
                    "safe_selector_choice": item.safe_selector_choice,
                    "safe_selector_reason": item.safe_selector_reason,
                    "safe_selector_local_objective": item.safe_selector_local_objective,
                    "safe_selector_joint_objective": item.safe_selector_joint_objective,
                    "predicted_p2_slot_count": int(item.predicted_p2_slot_count),
                    "bound_exact_p2_task_count": int(item.bound_exact_p2_task_count),
                    "unmatched_exact_p2_task_count": int(item.unmatched_exact_p2_task_count),
                    "exact_bind_count": int(item.exact_bind_count),
                    "boundary_mismatch_bind_count": int(item.boundary_mismatch_bind_count),
                    "overflow_bind_count": int(item.overflow_bind_count),
                    "unused_slot_count": int(item.unused_slot_count),
                    "appended_task_count": int(item.appended_task_count),
                    "repair_task_count": int(item.repair_task_count),
                    "repair_task_bytes": int(item.repair_task_bytes),
                    "repair_task_ratio_ppm": int(item.repair_task_ratio_ppm),
                    "repair_byte_ratio_ppm": int(item.repair_byte_ratio_ppm),
                    "binding_repair_reason": item.binding_repair_reason,
                    "prediction_fallback_reason": item.prediction_fallback_reason,
                    "prediction_generated": bool(item.prediction_generated),
                    "prediction_nonempty": bool(item.prediction_nonempty),
                    "prediction_validated": bool(item.prediction_validated),
                    "prediction_consumed": bool(item.prediction_consumed),
                    "prediction_fallback": bool(item.prediction_fallback),
                    "p0_p1_local_complete_times_ns": item.p0_p1_local_complete_times_ns,
                    "p0_p1_barrier_release_ns": item.p0_p1_barrier_release_ns,
                    "p0_p1_barrier_wait_ns_by_rank": item.p0_p1_barrier_wait_ns_by_rank,
                    "p0_p1_barrier_wait_ns_sum": int(item.p0_p1_barrier_wait_ns_sum),
                    "p0_p1_barrier_wait_ns_max": int(item.p0_p1_barrier_wait_ns_max),
                }
            )
        run_forward_makespan_ns = max((int(item.run_forward_makespan_ns) for item in formal), default=0)
        row = {
            "schema_version": "RS_SIM_EXPERIMENT_RUN_RESULT",
            "status": "PASS",
            "warmup": warmup,
            "run_id": run_id,
            "simulation_run_id": simulation_run_id,
            "paired_instance_id": paired_instance_id,
            "fixture_index": int(fixture_index),
            "treatment_index": int(treatment_index),
            "repeat_index": int(repeat_index),
            "fixture_path": str(fixture_path),
            "fixture_id": fixture.fixture_id,
            "fixture_truth_digest": fixture_truth_digest,
            "fixture_validation_mode": validation_mode,
            "fixture_validation_elapsed_ns": int(fixture_validation_elapsed_ns),
            "world_size": fixture.world_size,
            **_trace_identity_fields(fixture_path, fixture),
            "treatment": treatment,
            "oracle_solver_records": tuple(
                record
                for record in evidence.get("scheduler_algorithm_diagnostics", ())
                if "oracle_solver_status" in dict(record.get("diagnostics", {}))
            ),
            "timestamps_processed": timestamps,
            "window_count": len(windows),
            "max_task_bytes": int(config["simulation"]["max_task_bytes"]),
            "max_window_prefix_tasks": int(config["simulation"]["max_window_prefix_tasks"]),
            "p0_p1_compute_end_barrier": bool(config["simulation"]["p0_p1_compute_end_barrier"]),
            "p0_p1_barrier_wait_ns_sum": sum(int(item.p0_p1_barrier_wait_ns_sum) for item in windows),
            "p0_p1_barrier_wait_ns_max": max((int(item.p0_p1_barrier_wait_ns_max) for item in windows), default=0),
            "p0_p1_barrier_wait_ns_by_window": [
                tuple(int(value) for value in item.p0_p1_barrier_wait_ns_by_rank)
                for item in windows
            ],
            "window_makespan_ns_values": window_makespans,
            "window_makespan_ns_sum": sum(window_makespans),
            "window_makespan_ns_mean": int(round(statistics.mean(window_makespans))) if window_makespans else 0,
            "communication_metric_primary": "mean_communication_stall_ns",
            "communication_stall_metric_status": "AVAILABLE_CAUSAL_RANK_EXPOSURE",
            "communication_stall_metric_definition": (
                "P1(destination_compute_ready-source_local_path_complete)+"
                "P2(all_inbound_assembled-model_thread_ready)"
            ),
            "communication_stall_metric_compute_excluded": True,
            "communication_stall_metric_full_zero_comm_dual_execution": False,
            "formal_zero_comm_supported": False,
            "formal_communication_induced_delay_ns": None,
            "formal_communication_induced_delay_status": "BLOCKED_NO_FROZEN_PREPARED_PLAN_DUAL_EXECUTION",
            "network_transfer_span_ns_values": network_transfer_spans,
            "network_transfer_span_ns_sum": sum(network_transfer_spans),
            "compute_excluded_communication_makespan_ns_values": compute_excluded_communication_makespans,
            "compute_excluded_communication_makespan_ns_sum": sum(compute_excluded_communication_makespans),
            "compute_excluded_communication_makespan_ns_mean": (
                int(round(statistics.mean(compute_excluded_communication_makespans)))
                if compute_excluded_communication_makespans else 0
            ),
            "network_active_union_ns_values": network_active_unions,
            "network_active_union_ns_sum": sum(network_active_unions),
            "rank_communication_exposed_ns_by_window": [
                tuple(int(value) for value in item.rank_communication_exposed_ns_by_rank)
                for item in windows
            ],
            "rank_communication_exposed_ns_values": rank_communication_exposed_all_values,
            "rank_communication_exposed_ns_sum": sum(rank_communication_exposed_all_values),
            "rank_communication_exposed_ns_mean": (
                int(round(statistics.mean(rank_communication_exposed_all_values)))
                if rank_communication_exposed_all_values else 0
            ),
            "rank_communication_exposed_ns_max": max(rank_communication_exposed_all_values, default=0),
            "rank_communication_exposed_ns_p95": _int_percentile(rank_communication_exposed_all_values, 95),
            "rank_communication_exposed_ns_p99": _int_percentile(rank_communication_exposed_all_values, 99),
            "communication_stall_ns_by_window": [
                tuple(int(value) for value in item.rank_communication_exposed_ns_by_rank)
                for item in windows
            ],
            "communication_stall_ns_values": rank_communication_exposed_all_values,
            "mean_communication_stall_ns": (
                int(round(statistics.mean(rank_communication_exposed_all_values)))
                if rank_communication_exposed_all_values else 0
            ),
            "p95_communication_stall_ns": _int_percentile(rank_communication_exposed_all_values, 95),
            "max_communication_stall_ns": max(rank_communication_exposed_all_values, default=0),
            "rank_communication_exposed_window_max_ns_values": rank_communication_exposed_max_values,
            "rank_communication_exposed_window_spread_ns_values": rank_communication_exposed_window_spreads,
            "rank_communication_exposed_window_spread_ns_mean": int(round(statistics.mean(rank_communication_exposed_window_spreads))) if rank_communication_exposed_window_spreads else 0,
            "rank_communication_exposed_window_spread_ns_p95": _int_percentile(rank_communication_exposed_window_spreads, 95),
            "rank_communication_exposed_window_spread_ns_p99": _int_percentile(rank_communication_exposed_window_spreads, 99),
            "rank_communication_exposed_window_spread_ns_max": max(rank_communication_exposed_window_spreads, default=0),
            "p0_p1_local_completion_spread_ns_values": p0_p1_local_completion_spreads,
            "p0_p1_local_completion_spread_ns_max": max(p0_p1_local_completion_spreads, default=0),
            "run_forward_makespan_ns": run_forward_makespan_ns,
            "ttft_proxy_ns": run_forward_makespan_ns,
            "ttft_metric_scope": "MOE_FORWARD_ONLY_PROXY_NOT_SERVICE_TTFT",
            "ttft_claim_allowed": False,
            "p2_first_rank_release_offset_ns_values": p2_first_release_offsets,
            "p2_last_rank_release_offset_ns_values": p2_last_release_offsets,
            "p2_rank_release_spread_ns_values": p2_release_spreads,
            "p2_rank_release_spread_ns_max": max(p2_release_spreads, default=0),
            "p2_all_rank_release_offset_ns_values": all_p2_release_offsets,
            "completed_bytes": sum(int(item.completed_bytes) for item in formal),
            "receiver_total_delay_ns": sum(int(item.receiver_total_delay_ns) for item in formal),
            "receiver_posting_queue_wait_ns": sum(
                int(item.receiver_posting_queue_wait_ns) for item in formal
            ),
            "receiver_buffer_stall_ns": sum(
                int(item.receiver_buffer_stall_ns) for item in formal
            ),
            "receiver_posting_service_ns": sum(
                int(item.receiver_posting_service_ns) for item in formal
            ),
            "receiver_drain_queue_wait_ns": sum(
                int(item.receiver_drain_queue_wait_ns) for item in formal
            ),
            "receiver_drain_service_ns": sum(
                int(item.receiver_drain_service_ns) for item in formal
            ),
            "prediction_hidden_ns": sum(int(item.prediction_hidden_ns) for item in formal),
            "prediction_exposed_ns": sum(int(item.prediction_exposed_ns) for item in formal),
            "control_hidden_ns": sum(int(item.control_hidden_ns) for item in formal),
            "control_exposed_ns": sum(int(item.control_exposed_ns) for item in formal),
            "binding_hidden_ns": sum(int(item.binding_hidden_ns) for item in formal),
            "binding_exposed_ns": sum(int(item.binding_exposed_ns) for item in formal),
            "prediction_absolute_error_bytes": sum(int(item.prediction_absolute_error_bytes or 0) for item in windows),
            "prediction_relative_absolute_error_ppm_values": prediction_relative_errors,
            "prediction_relative_absolute_error_ppm_mean": int(round(statistics.mean(prediction_relative_errors))) if prediction_relative_errors else 0,
            "prediction_matrix_overlap_ppm_values": prediction_overlaps,
            "prediction_matrix_overlap_ppm_mean": int(round(statistics.mean(prediction_overlaps))) if prediction_overlaps else 0,
            "prediction_top_destination_accuracy_ppm_values": prediction_top_destination,
            "prediction_top_destination_accuracy_ppm_mean": int(round(statistics.mean(prediction_top_destination))) if prediction_top_destination else 0,
            "task_catalogue_digests": sorted({item.task_catalogue_digest for item in windows}),
            "task_boundary_digests": sorted({item.task_boundary_digest for item in windows}),
            "window_truth_digests": [item.truth_digest for item in windows],
            "template_digests": [item.template_digest for item in windows],
            "algorithm_core_run_count": sum(int(item.algorithm_core_run_count) for item in windows),
            "incremental_bind_job_count": sum(int(item.incremental_bind_job_count) for item in windows),
            "safe_selector_choices": [item.safe_selector_choice for item in windows],
            "safe_selector_reasons": [item.safe_selector_reason for item in windows],
            "algorithm_identity": str(treatment.get("algorithm", "")),
            "global_safe_selector_enabled": bool(treatment.get("safe_selector", False)),
            "exact_bind_count": sum(int(item.exact_bind_count) for item in windows),
            "boundary_mismatch_bind_count": sum(int(item.boundary_mismatch_bind_count) for item in windows),
            "overflow_bind_count": sum(int(item.overflow_bind_count) for item in windows),
            "unused_slot_count": sum(int(item.unused_slot_count) for item in windows),
            "appended_task_count": sum(int(item.appended_task_count) for item in windows),
            "repair_task_count": sum(int(item.repair_task_count) for item in windows),
            "repair_task_bytes": sum(int(item.repair_task_bytes) for item in windows),
            "repair_task_ratio_ppm_values": [int(item.repair_task_ratio_ppm) for item in windows],
            "repair_byte_ratio_ppm_values": [int(item.repair_byte_ratio_ppm) for item in windows],
            "binding_repair_reasons": [item.binding_repair_reason for item in windows],
            "prediction_fallback_reasons": [item.prediction_fallback_reason for item in windows],
            "prediction_generated_count": sum(bool(item.prediction_generated) for item in windows),
            "prediction_nonempty_count": sum(bool(item.prediction_nonempty) for item in windows),
            "prediction_validated_count": sum(bool(item.prediction_validated) for item in windows),
            "prediction_consumed_count": sum(bool(item.prediction_consumed) for item in windows),
            "prediction_fallback_count": sum(bool(item.prediction_fallback) for item in windows),
            "physical_transfer_digest": stable_digest(_freeze(timeline)),
            "performance_claim_allowed": False,
            "hardware_profile_calibrated": False,
            "per_window_metrics": per_window_metrics,
            "wall_clock_ns": int(time.monotonic_ns() - wall_started_ns),
        }
        if config["save_plans"]:
            row["anchor_window_evidence"] = [dataclasses.asdict(item) for item in windows]
            row["formal_runtime_records"] = [dataclasses.asdict(item) for item in formal]
        if config["save_task_timeline"]:
            row["transfer_timeline"] = timeline
        if config["save_raw_events"]:
            row["runtime_evidence"] = evidence
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "result.json"
        path.write_text(stable_json_dumps(_freeze(row)) + "\n", encoding="utf-8")
        row["result_path"] = str(path)
        return row
    finally:
        if dispose_runtime:
            runtime.dispose()



def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def _csv_scalar(value: Any) -> Any:
    """Return a stable CSV cell without discarding structured evidence."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(
        _freeze(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _flatten_csv_mapping(
    value: dict[str, Any],
    *,
    prefix: str = "",
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten mappings; preserve sequences and complex values as JSON cells."""
    result = {} if target is None else target
    for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        key = str(raw_key)
        column = f"{prefix}__{key}" if prefix else key
        if isinstance(item, dict):
            _flatten_csv_mapping(item, prefix=column, target=result)
        else:
            result[column] = _csv_scalar(item)
    return result


def _complete_csv_row(
    row: dict[str, Any],
    *,
    record_type: str,
    normalized: dict[str, Any],
) -> dict[str, Any]:
    treatment = row.get("treatment") if isinstance(row.get("treatment"), dict) else {}
    source = normalized.get("__source") if isinstance(normalized.get("__source"), dict) else {}
    result: dict[str, Any] = {
        "record_type": str(record_type),
        "status": str(row.get("status", "PASS")),
        "experiment_name": str(normalized["name"]),
        "config_path": str(source.get("__config_path", "")),
        "trace_roots": _csv_scalar([str(path) for path in normalized["trace_paths"]]),
        "fixture_index": row.get("fixture_index"),
        "fixture_id": row.get("fixture_id"),
        "fixture_path": row.get("fixture_path"),
        "trace_mode": row.get("trace_mode"),
        "trace_model": row.get("trace_model"),
        "trace_ep": row.get("trace_ep"),
        "trace_sequence_length": row.get("trace_sequence_length"),
        "world_size": row.get("world_size"),
        "repeat_index": row.get("repeat_index"),
        "warmup": row.get("warmup"),
        "treatment_name": treatment.get("name"),
        "algorithm_core": treatment.get("core"),
        "algorithm_policy": treatment.get("policy"),
        "scope": treatment.get("scope"),
        "planning": treatment.get("planning"),
        "information": treatment.get("information"),
        "overlap": treatment.get("overlap"),
        "experiment_role": treatment.get("experiment_role"),
    }
    for section_name in ("simulation", "repetitions", "execution", "oracle", "comparison"):
        section = normalized.get(section_name)
        if isinstance(section, dict):
            _flatten_csv_mapping(section, prefix=f"config__{section_name}", target=result)
    flattened = _flatten_csv_mapping(row)
    for key, item in flattened.items():
        result.setdefault(key, item)
    return result


def _complete_csv_rows(
    runtime_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    *,
    normalized: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _complete_csv_row(row, record_type="RUNTIME", normalized=normalized)
        for row in runtime_rows
    ]
    rows.extend(
        _complete_csv_row(row, record_type="ORACLE", normalized=normalized)
        for row in oracle_rows
    )
    rows.extend(
        _complete_csv_row(row, record_type="FAILURE", normalized=normalized)
        for row in statuses
        if str(row.get("status", "")).upper() == "FAILED"
    )
    return rows


def _write_complete_results_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    priority = [
        "record_type",
        "status",
        "experiment_name",
        "config_path",
        "trace_roots",
        "fixture_index",
        "fixture_id",
        "fixture_path",
        "trace_mode",
        "trace_model",
        "trace_ep",
        "trace_sequence_length",
        "world_size",
        "repeat_index",
        "warmup",
        "treatment_name",
        "algorithm_core",
        "algorithm_policy",
        "scope",
        "planning",
        "information",
        "overlap",
        "experiment_role",
    ]
    fields = set()
    for row in rows:
        fields.update(row)
    ordered = [field for field in priority if field in fields]
    ordered.extend(sorted(fields.difference(ordered)))
    _write_csv(path, rows, ordered or ["record_type"])


def _percentile(values: Iterable[int | float], percentile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    if len(rows) == 1:
        return rows[0]
    q = max(0.0, min(100.0, float(percentile))) / 100.0
    position = q * (len(rows) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return rows[lower]
    fraction = position - lower
    return rows[lower] * (1.0 - fraction) + rows[upper] * fraction


def _int_percentile(values: Iterable[int], percentile: float) -> int:
    return int(round(_percentile(values, percentile)))


def _metric_summary(prefix: str, values: Iterable[int]) -> dict[str, int]:
    rows = [int(value) for value in values]
    if not rows:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": 0,
            f"{prefix}_p50": 0,
            f"{prefix}_p95": 0,
            f"{prefix}_p99": 0,
            f"{prefix}_min": 0,
            f"{prefix}_max": 0,
        }
    return {
        f"{prefix}_count": len(rows),
        f"{prefix}_mean": int(round(statistics.mean(rows))),
        f"{prefix}_p50": _int_percentile(rows, 50),
        f"{prefix}_p95": _int_percentile(rows, 95),
        f"{prefix}_p99": _int_percentile(rows, 99),
        f"{prefix}_min": min(rows),
        f"{prefix}_max": max(rows),
    }


def _flatten_metric(rows: Iterable[dict[str, Any]], key: str) -> list[int]:
    result: list[int] = []
    for row in rows:
        value = row.get(key, [])
        if isinstance(value, (list, tuple)):
            result.extend(int(item) for item in value if item is not None)
        elif value is not None:
            result.append(int(value))
    return result


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        key = (row["fixture_id"], row["treatment"]["name"])
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (fixture_id, treatment_name), rows in sorted(grouped.items()):
        first = rows[0]
        run_windows = [int(item["window_makespan_ns_sum"]) for item in rows]
        run_forward = [int(item["run_forward_makespan_ns"]) for item in rows]
        run_network_transfer_span = [int(item["network_transfer_span_ns_sum"]) for item in rows]
        run_compute_excluded_communication = [
            int(item["compute_excluded_communication_makespan_ns_sum"]) for item in rows
        ]
        run_network_active_union = [int(item["network_active_union_ns_sum"]) for item in rows]
        wall_clock_us = [int(item.get("wall_clock_ns", 0)) // 1000 for item in rows]
        window_values = _flatten_metric(rows, "window_makespan_ns_values")
        network_transfer_span_values = _flatten_metric(rows, "network_transfer_span_ns_values")
        compute_excluded_communication_values = _flatten_metric(
            rows, "compute_excluded_communication_makespan_ns_values"
        )
        network_active_union_values = _flatten_metric(rows, "network_active_union_ns_values")
        rank_communication_exposed_values = _flatten_metric(
            rows, "rank_communication_exposed_ns_values"
        )
        rank_communication_exposed_window_max_values = _flatten_metric(
            rows, "rank_communication_exposed_window_max_ns_values"
        )
        first_release = _flatten_metric(rows, "p2_first_rank_release_offset_ns_values")
        last_release = _flatten_metric(rows, "p2_last_rank_release_offset_ns_values")
        all_release = _flatten_metric(rows, "p2_all_rank_release_offset_ns_values")
        prediction_error = _flatten_metric(rows, "prediction_relative_absolute_error_ppm_values")
        prediction_overlap = _flatten_metric(rows, "prediction_matrix_overlap_ppm_values")
        prediction_top = _flatten_metric(rows, "prediction_top_destination_accuracy_ppm_values")
        summary: dict[str, Any] = {
            "fixture_id": fixture_id,
            "world_size": first["world_size"],
            "trace_mode": first.get("trace_mode"),
            "trace_ep": first.get("trace_ep"),
            "trace_model": first.get("trace_model"),
            "trace_sequence_length": first.get("trace_sequence_length"),
            "trace_dataset_id": first.get("trace_dataset_id"),
            "trace_split": first.get("trace_split"),
            "trace_capture_id": first.get("trace_capture_id"),
            "trace_collector_version": first.get("trace_collector_version"),
            "trace_source_kind": first.get("trace_source_kind"),
            "trace_provenance_digest": first.get("trace_provenance_digest"),
            "treatment": treatment_name,
            "core": first["treatment"]["core"],
            "scope": first["treatment"]["scope"],
            "planning": first["treatment"]["planning"],
            "information": first["treatment"]["information"],
            "overlap": first["treatment"]["overlap"],
            "measure_count": len(rows),
            "completed_bytes": int(first["completed_bytes"]),
            "all_runs_terminal": all(item.get("status") == "PASS" for item in rows),
            "ttft_metric_scope": "MOE_FORWARD_ONLY_PROXY_NOT_SERVICE_TTFT",
            "ttft_claim_allowed": False,
            "prediction_exposed_ns_mean": int(round(statistics.mean(int(item["prediction_exposed_ns"]) for item in rows))),
            "control_exposed_ns_mean": int(round(statistics.mean(int(item["control_exposed_ns"]) for item in rows))),
            "binding_exposed_ns_mean": int(round(statistics.mean(int(item["binding_exposed_ns"]) for item in rows))),
            "prediction_relative_absolute_error_percent_mean": round(
                statistics.mean(prediction_error) / 10_000.0, 6
            ) if prediction_error else 0.0,
            "prediction_matrix_overlap_percent_mean": round(
                statistics.mean(prediction_overlap) / 10_000.0, 6
            ) if prediction_overlap else 0.0,
            "prediction_top_destination_accuracy_percent_mean": round(
                statistics.mean(prediction_top) / 10_000.0, 6
            ) if prediction_top else 0.0,
            "performance_claim_allowed": False,
            "hardware_profile_calibrated": False,
        }
        summary.update(_metric_summary("run_window_makespan_ns", run_windows))
        summary.update(_metric_summary("run_forward_makespan_ns", run_forward))
        summary.update(_metric_summary("ttft_proxy_ns", run_forward))
        summary.update(_metric_summary("run_network_transfer_span_ns", run_network_transfer_span))
        summary.update(_metric_summary(
            "run_compute_excluded_communication_makespan_ns",
            run_compute_excluded_communication,
        ))
        summary.update(_metric_summary("run_network_active_union_ns", run_network_active_union))
        summary.update(_metric_summary("window_makespan_ns", window_values))
        summary.update(_metric_summary("network_transfer_span_ns", network_transfer_span_values))
        summary.update(_metric_summary(
            "compute_excluded_communication_makespan_ns",
            compute_excluded_communication_values,
        ))
        summary.update(_metric_summary("network_active_union_ns", network_active_union_values))
        summary.update(_metric_summary(
            "rank_communication_exposed_ns", rank_communication_exposed_values
        ))
        summary.update(_metric_summary(
            "communication_stall_ns", rank_communication_exposed_values
        ))
        summary.update(_metric_summary(
            "rank_communication_exposed_window_max_ns",
            rank_communication_exposed_window_max_values,
        ))
        summary.update(_metric_summary("p2_first_rank_release_offset_ns", first_release))
        summary.update(_metric_summary("p2_last_rank_release_offset_ns", last_release))
        summary.update(_metric_summary("p2_all_rank_release_offset_ns", all_release))
        summary.update(_metric_summary("worker_wall_clock_us", wall_clock_us))
        summaries.append(summary)
    return summaries


def _improvement_percent(baseline: int, candidate: int) -> float:
    if baseline <= 0:
        return 0.0
    return (float(baseline) - float(candidate)) * 100.0 / float(baseline)


def _paired_rows(
    results: list[dict[str, Any]],
    *,
    baseline_name: str,
    baselines_by_overlap: dict[str, str] | None = None,
    reference_baselines: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_instance: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in results:
        key = (str(row["fixture_id"]), int(row["repeat_index"]))
        by_instance.setdefault(key, {})[str(row["treatment"]["name"])] = row
    run_pairs: list[dict[str, Any]] = []
    window_pairs: list[dict[str, Any]] = []
    explicit_overlap_map = dict(baselines_by_overlap or {})
    requested_references = tuple(dict.fromkeys((baseline_name, *reference_baselines)))
    for (fixture_id, repeat_index), rows in sorted(by_instance.items()):
        references_by_overlap: dict[str, list[str]] = {}
        for reference_name in requested_references:
            reference = rows.get(reference_name)
            if reference is None:
                continue
            references_by_overlap.setdefault(
                str(reference["treatment"]["overlap"]), []
            ).append(reference_name)
        for overlap_mode, reference_name in explicit_overlap_map.items():
            if reference_name in rows:
                bucket = references_by_overlap.setdefault(str(overlap_mode), [])
                if reference_name not in bucket:
                    bucket.insert(0, reference_name)
        for candidate_name, candidate in sorted(rows.items()):
            candidate_overlap = str(candidate["treatment"]["overlap"])
            for selected_baseline_name in references_by_overlap.get(candidate_overlap, []):
                if candidate_name == selected_baseline_name:
                    continue
                baseline = rows[selected_baseline_name]
                if str(baseline["treatment"]["overlap"]) != candidate_overlap:
                    continue
                baseline_windows = {
                    (int(item["anchor_layer_id"]), str(item["truth_digest"])): item
                    for item in baseline.get("per_window_metrics", [])
                }
                fairness = {
                    "same_overlap_mode": str(baseline["treatment"]["overlap"]) == candidate_overlap,
                    "same_p0_p1_compute_end_barrier": bool(baseline.get("p0_p1_compute_end_barrier", False)) == bool(candidate.get("p0_p1_compute_end_barrier", False)),
                    "same_fixture_truth_digest": baseline["fixture_truth_digest"] == candidate["fixture_truth_digest"],
                    "same_task_catalogue_digest": baseline["task_catalogue_digests"] == candidate["task_catalogue_digests"],
                    "same_task_boundary_digest": baseline["task_boundary_digests"] == candidate["task_boundary_digests"],
                    "same_window_truth": baseline["window_truth_digests"] == candidate["window_truth_digests"],
                    "same_completed_bytes": int(baseline["completed_bytes"]) == int(candidate["completed_bytes"]),
                }
                run_pairs.append(
                    {
                        "fixture_id": fixture_id,
                        "trace_mode": baseline.get("trace_mode"),
                        "trace_ep": baseline.get("trace_ep"),
                        "trace_model": baseline.get("trace_model"),
                        "trace_sequence_length": baseline.get("trace_sequence_length"),
                        "trace_dataset_id": baseline.get("trace_dataset_id"),
                        "trace_split": baseline.get("trace_split"),
                        "trace_capture_id": baseline.get("trace_capture_id"),
                        "trace_source_kind": baseline.get("trace_source_kind"),
                        "trace_provenance_digest": baseline.get("trace_provenance_digest"),
                        "repeat_index": repeat_index,
                        "paired_instance_id": baseline["paired_instance_id"],
                        "baseline": selected_baseline_name,
                        "candidate": candidate_name,
                        "overlap_mode": candidate_overlap,
                        "p0_p1_compute_end_barrier": bool(candidate.get("p0_p1_compute_end_barrier", False)),
                        "baseline_window_makespan_ns": int(baseline["window_makespan_ns_sum"]),
                        "candidate_window_makespan_ns": int(candidate["window_makespan_ns_sum"]),
                        "window_improvement_percent": round(_improvement_percent(int(baseline["window_makespan_ns_sum"]), int(candidate["window_makespan_ns_sum"])), 6),
                        "baseline_network_transfer_span_ns": int(baseline["network_transfer_span_ns_sum"]),
                        "candidate_network_transfer_span_ns": int(candidate["network_transfer_span_ns_sum"]),
                        "network_transfer_span_improvement_percent": round(_improvement_percent(int(baseline["network_transfer_span_ns_sum"]), int(candidate["network_transfer_span_ns_sum"])), 6),
                        "baseline_compute_excluded_communication_makespan_ns": int(baseline["compute_excluded_communication_makespan_ns_sum"]),
                        "candidate_compute_excluded_communication_makespan_ns": int(candidate["compute_excluded_communication_makespan_ns_sum"]),
                        "compute_excluded_communication_improvement_percent": round(
                            _improvement_percent(
                                int(baseline["compute_excluded_communication_makespan_ns_sum"]),
                                int(candidate["compute_excluded_communication_makespan_ns_sum"]),
                            ),
                            6,
                        ),
                        "baseline_mean_communication_stall_ns": int(baseline["mean_communication_stall_ns"]),
                        "candidate_mean_communication_stall_ns": int(candidate["mean_communication_stall_ns"]),
                        "mean_communication_stall_improvement_percent": round(
                            _improvement_percent(
                                int(baseline["mean_communication_stall_ns"]),
                                int(candidate["mean_communication_stall_ns"]),
                            ),
                            6,
                        ),
                        "baseline_p95_communication_stall_ns": int(baseline["p95_communication_stall_ns"]),
                        "candidate_p95_communication_stall_ns": int(candidate["p95_communication_stall_ns"]),
                        "p95_communication_stall_improvement_percent": round(
                            _improvement_percent(
                                int(baseline["p95_communication_stall_ns"]),
                                int(candidate["p95_communication_stall_ns"]),
                            ),
                            6,
                        ),
                        "baseline_rank_communication_exposed_ns_max": int(baseline["rank_communication_exposed_ns_max"]),
                        "candidate_rank_communication_exposed_ns_max": int(candidate["rank_communication_exposed_ns_max"]),
                        "rank_communication_exposed_max_improvement_percent": round(
                            _improvement_percent(
                                int(baseline["rank_communication_exposed_ns_max"]),
                                int(candidate["rank_communication_exposed_ns_max"]),
                            ),
                            6,
                        ),
                        "baseline_run_forward_makespan_ns": int(baseline["run_forward_makespan_ns"]),
                        "candidate_run_forward_makespan_ns": int(candidate["run_forward_makespan_ns"]),
                        "run_forward_improvement_percent": round(_improvement_percent(int(baseline["run_forward_makespan_ns"]), int(candidate["run_forward_makespan_ns"])), 6),
                        "baseline_ttft_proxy_ns": int(baseline["ttft_proxy_ns"]),
                        "candidate_ttft_proxy_ns": int(candidate["ttft_proxy_ns"]),
                        "ttft_proxy_improvement_percent": round(_improvement_percent(int(baseline["ttft_proxy_ns"]), int(candidate["ttft_proxy_ns"])), 6),
                        "fairness_pass": all(fairness.values()),
                        **fairness,
                    }
                )
                candidate_windows = {
                    (int(item["anchor_layer_id"]), str(item["truth_digest"])): item
                    for item in candidate.get("per_window_metrics", [])
                }
                for key, base_window in sorted(baseline_windows.items()):
                    cand_window = candidate_windows.get(key)
                    if cand_window is None:
                        continue
                    base_first = base_window.get("p2_first_rank_release_offset_ns")
                    cand_first = cand_window.get("p2_first_rank_release_offset_ns")
                    base_last = base_window.get("p2_last_rank_release_offset_ns")
                    cand_last = cand_window.get("p2_last_rank_release_offset_ns")
                    window_pairs.append(
                        {
                            "fixture_id": fixture_id,
                            "trace_mode": baseline.get("trace_mode"),
                            "trace_ep": baseline.get("trace_ep"),
                            "trace_model": baseline.get("trace_model"),
                            "trace_sequence_length": baseline.get("trace_sequence_length"),
                            "trace_dataset_id": baseline.get("trace_dataset_id"),
                            "trace_split": baseline.get("trace_split"),
                            "trace_capture_id": baseline.get("trace_capture_id"),
                            "trace_source_kind": baseline.get("trace_source_kind"),
                            "trace_provenance_digest": baseline.get("trace_provenance_digest"),
                            "repeat_index": repeat_index,
                            "anchor_layer_id": key[0],
                            "window_truth_digest": key[1],
                            "baseline": selected_baseline_name,
                            "candidate": candidate_name,
                            "overlap_mode": candidate_overlap,
                            "baseline_window_makespan_ns": int(base_window["window_makespan_ns"]),
                            "candidate_window_makespan_ns": int(cand_window["window_makespan_ns"]),
                            "window_improvement_percent": round(_improvement_percent(int(base_window["window_makespan_ns"]), int(cand_window["window_makespan_ns"])), 6),
                            "baseline_network_transfer_span_ns": int(base_window["network_transfer_span_ns"]),
                            "candidate_network_transfer_span_ns": int(cand_window["network_transfer_span_ns"]),
                            "network_transfer_span_improvement_percent": round(_improvement_percent(int(base_window["network_transfer_span_ns"]), int(cand_window["network_transfer_span_ns"])), 6),
                            "baseline_compute_excluded_communication_makespan_ns": int(base_window["compute_excluded_communication_makespan_ns"]),
                            "candidate_compute_excluded_communication_makespan_ns": int(cand_window["compute_excluded_communication_makespan_ns"]),
                            "compute_excluded_communication_improvement_percent": round(
                                _improvement_percent(
                                    int(base_window["compute_excluded_communication_makespan_ns"]),
                                    int(cand_window["compute_excluded_communication_makespan_ns"]),
                                ),
                                6,
                            ),
                            "baseline_mean_communication_stall_ns": int(base_window["mean_communication_stall_ns"]),
                            "candidate_mean_communication_stall_ns": int(cand_window["mean_communication_stall_ns"]),
                            "mean_communication_stall_improvement_percent": round(
                                _improvement_percent(
                                    int(base_window["mean_communication_stall_ns"]),
                                    int(cand_window["mean_communication_stall_ns"]),
                                ),
                                6,
                            ),
                            "baseline_p95_communication_stall_ns": int(base_window["p95_communication_stall_ns"]),
                            "candidate_p95_communication_stall_ns": int(cand_window["p95_communication_stall_ns"]),
                            "p95_communication_stall_improvement_percent": round(
                                _improvement_percent(
                                    int(base_window["p95_communication_stall_ns"]),
                                    int(cand_window["p95_communication_stall_ns"]),
                                ),
                                6,
                            ),
                            "baseline_rank_communication_exposed_ns_max": int(base_window["rank_communication_exposed_ns_max"]),
                            "candidate_rank_communication_exposed_ns_max": int(cand_window["rank_communication_exposed_ns_max"]),
                            "rank_communication_exposed_max_improvement_percent": round(
                                _improvement_percent(
                                    int(base_window["rank_communication_exposed_ns_max"]),
                                    int(cand_window["rank_communication_exposed_ns_max"]),
                                ),
                                6,
                            ),
                            "baseline_p2_first_rank_release_offset_ns": base_first,
                            "candidate_p2_first_rank_release_offset_ns": cand_first,
                            "p2_first_rank_release_improvement_percent": (
                                None if base_first is None or cand_first is None
                                else round(_improvement_percent(int(base_first), int(cand_first)), 6)
                            ),
                            "baseline_p2_last_rank_release_offset_ns": base_last,
                            "candidate_p2_last_rank_release_offset_ns": cand_last,
                            "p2_last_rank_release_improvement_percent": (
                                None if base_last is None or cand_last is None
                                else round(_improvement_percent(int(base_last), int(cand_last)), 6)
                            ),
                            "same_overlap_mode": True,
                            "same_p0_p1_compute_end_barrier": bool(baseline.get("p0_p1_compute_end_barrier", False)) == bool(candidate.get("p0_p1_compute_end_barrier", False)),
                            "same_task_catalogue_digest": base_window["task_catalogue_digest"] == cand_window["task_catalogue_digest"],
                            "same_task_boundary_digest": base_window["task_boundary_digest"] == cand_window["task_boundary_digest"],
                        }
                    )
    return run_pairs, window_pairs


def _paired_summary_rows(
    run_pairs: list[dict[str, Any]],
    window_pairs: list[dict[str, Any]],
    *,
    target_improvement_percent: float,
    tie_tolerance_percent: float,
) -> list[dict[str, Any]]:
    grouped_runs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    grouped_windows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in run_pairs:
        grouped_runs.setdefault((str(row["baseline"]), str(row["candidate"])), []).append(row)
    for row in window_pairs:
        grouped_windows.setdefault((str(row["baseline"]), str(row["candidate"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped_runs):
        runs = grouped_runs[key]
        windows = grouped_windows.get(key, [])
        window_improvements = [float(row["window_improvement_percent"]) for row in windows]
        compute_excluded_window_improvements = [
            float(row["compute_excluded_communication_improvement_percent"])
            for row in windows
        ]
        mean_stall_window_improvements = [
            float(row["mean_communication_stall_improvement_percent"])
            for row in windows
        ]
        p95_stall_window_improvements = [
            float(row["p95_communication_stall_improvement_percent"])
            for row in windows
        ]
        rank_exposed_window_improvements = [
            float(row["rank_communication_exposed_max_improvement_percent"])
            for row in windows
        ]
        run_improvements = [float(row["window_improvement_percent"]) for row in runs]
        communication_improvements = [
            float(row["network_transfer_span_improvement_percent"]) for row in runs
        ]
        compute_excluded_communication_improvements = [
            float(row["compute_excluded_communication_improvement_percent"]) for row in runs
        ]
        mean_stall_improvements = [
            float(row["mean_communication_stall_improvement_percent"]) for row in runs
        ]
        p95_stall_improvements = [
            float(row["p95_communication_stall_improvement_percent"]) for row in runs
        ]
        rank_exposed_improvements = [
            float(row["rank_communication_exposed_max_improvement_percent"]) for row in runs
        ]
        forward_improvements = [
            float(row["run_forward_improvement_percent"]) for row in runs
        ]
        wins = sum(value > tie_tolerance_percent for value in window_improvements)
        losses = sum(value < -tie_tolerance_percent for value in window_improvements)
        ties = len(window_improvements) - wins - losses
        communication_only_wins = sum(
            value > tie_tolerance_percent for value in compute_excluded_window_improvements
        )
        communication_only_losses = sum(
            value < -tie_tolerance_percent for value in compute_excluded_window_improvements
        )
        communication_only_ties = (
            len(compute_excluded_window_improvements)
            - communication_only_wins
            - communication_only_losses
        )
        target_hits = sum(value >= target_improvement_percent for value in window_improvements)
        summary: dict[str, Any] = {
            "baseline": key[0],
            "candidate": key[1],
            "paired_run_count": len(runs),
            "paired_window_count": len(windows),
            "fairness_pass": all(bool(row["fairness_pass"]) for row in runs),
            "win_count": wins,
            "tie_count": ties,
            "loss_count": losses,
            "win_rate_percent": round(100.0 * wins / len(window_improvements), 6) if window_improvements else 0.0,
            "regression_rate_percent": round(100.0 * losses / len(window_improvements), 6) if window_improvements else 0.0,
            "compute_excluded_communication_win_count": communication_only_wins,
            "compute_excluded_communication_tie_count": communication_only_ties,
            "compute_excluded_communication_loss_count": communication_only_losses,
            "compute_excluded_communication_win_rate_percent": (
                round(100.0 * communication_only_wins / len(compute_excluded_window_improvements), 6)
                if compute_excluded_window_improvements else 0.0
            ),
            "compute_excluded_communication_improvement_mean_percent": (
                round(statistics.mean(compute_excluded_window_improvements), 6)
                if compute_excluded_window_improvements else 0.0
            ),
            "compute_excluded_communication_improvement_window_p50_percent": round(
                _percentile(compute_excluded_window_improvements, 50), 6
            ),
            "compute_excluded_communication_improvement_p95_percent": round(
                _percentile(compute_excluded_window_improvements, 95), 6
            ),
            "mean_communication_stall_improvement_window_p50_percent": round(
                _percentile(mean_stall_window_improvements, 50), 6
            ),
            "mean_communication_stall_improvement_window_p95_percent": round(
                _percentile(mean_stall_window_improvements, 95), 6
            ),
            "p95_communication_stall_improvement_window_p50_percent": round(
                _percentile(p95_stall_window_improvements, 50), 6
            ),
            "rank_communication_exposed_max_improvement_window_p50_percent": round(
                _percentile(rank_exposed_window_improvements, 50), 6
            ),
            "target_improvement_percent": float(target_improvement_percent),
            "target_hit_rate_percent": round(100.0 * target_hits / len(window_improvements), 6) if window_improvements else 0.0,
            "window_improvement_mean_percent": round(statistics.mean(window_improvements), 6) if window_improvements else 0.0,
            "window_improvement_p5_percent": round(_percentile(window_improvements, 5), 6),
            "window_improvement_p50_percent": round(_percentile(window_improvements, 50), 6),
            "window_improvement_p95_percent": round(_percentile(window_improvements, 95), 6),
            "window_improvement_p99_percent": round(_percentile(window_improvements, 99), 6),
            "run_window_improvement_p50_percent": round(_percentile(run_improvements, 50), 6),
            "run_window_improvement_p95_percent": round(_percentile(run_improvements, 95), 6),
            "communication_improvement_p50_percent": round(_percentile(communication_improvements, 50), 6),
            "compute_excluded_communication_improvement_p50_percent": round(
                _percentile(compute_excluded_communication_improvements, 50), 6
            ),
            "mean_communication_stall_improvement_p50_percent": round(
                _percentile(mean_stall_improvements, 50), 6
            ),
            "p95_communication_stall_improvement_p50_percent": round(
                _percentile(p95_stall_improvements, 50), 6
            ),
            "rank_communication_exposed_max_improvement_p50_percent": round(
                _percentile(rank_exposed_improvements, 50), 6
            ),
            "forward_improvement_p50_percent": round(_percentile(forward_improvements, 50), 6),
            "synthetic_30_percent_target_met": bool(
                window_improvements
                and all(bool(row["fairness_pass"]) for row in runs)
                and _percentile(window_improvements, 50) >= target_improvement_percent
                and losses == 0
            ),
            "paper_performance_claim_allowed": False,
            "paper_evidence_status": "MECHANISM_ONLY_UNCALIBRATED",
        }
        summaries.append(summary)
    return summaries


_GROUPING_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("overall", ()),
    ("trace_mode", ("trace_mode",)),
    ("ep", ("trace_mode", "trace_ep")),
    ("model", ("trace_mode", "trace_model")),
    ("model_ep", ("trace_mode", "trace_model", "trace_ep")),
    (
        "model_ep_sequence",
        ("trace_mode", "trace_model", "trace_ep", "trace_sequence_length"),
    ),
)


def _group_values(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def _grouped_paired_summary_rows(
    run_pairs: list[dict[str, Any]],
    window_pairs: list[dict[str, Any]],
    *,
    target_improvement_percent: float,
    tie_tolerance_percent: float,
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for group_kind, fields in _GROUPING_SPECS:
        run_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        window_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in run_pairs:
            run_groups.setdefault(_group_values(row, fields), []).append(row)
        for row in window_pairs:
            window_groups.setdefault(_group_values(row, fields), []).append(row)
        for values in sorted(run_groups, key=lambda item: tuple("" if v is None else str(v) for v in item)):
            summaries = _paired_summary_rows(
                run_groups[values],
                window_groups.get(values, []),
                target_improvement_percent=target_improvement_percent,
                tie_tolerance_percent=tie_tolerance_percent,
            )
            coordinates = {field: value for field, value in zip(fields, values)}
            for summary in summaries:
                grouped.append({
                    "group_kind": group_kind,
                    "trace_mode": coordinates.get("trace_mode"),
                    "trace_model": coordinates.get("trace_model"),
                    "trace_ep": coordinates.get("trace_ep"),
                    "trace_sequence_length": coordinates.get("trace_sequence_length"),
                    **summary,
                })
    return grouped


def _grouped_oracle_reference_rows(
    oracle_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for group_kind, fields in _GROUPING_SPECS:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in oracle_results:
            key = (*_group_values(row, fields), str(row["treatment"]["name"]))
            buckets.setdefault(key, []).append(row)
        for key in sorted(buckets, key=lambda item: tuple("" if v is None else str(v) for v in item)):
            rows = buckets[key]
            values = key[:-1]
            treatment_name = str(key[-1])
            coordinates = {field: value for field, value in zip(fields, values)}
            objective_sum = sum(int(row["objective_sum"]) for row in rows)
            fifo_sum = sum(int(row["logical_fifo_objective_sum"]) for row in rows)
            relative_ppm = (
                None
                if fifo_sum <= 0
                else int(round((fifo_sum - objective_sum) * 1_000_000 / fifo_sum))
            )
            window_count = sum(int(row["window_count"]) for row in rows)
            certified_count = sum(int(row["certified_window_count"]) for row in rows)
            grouped.append({
                "group_kind": group_kind,
                "trace_mode": coordinates.get("trace_mode"),
                "trace_model": coordinates.get("trace_model"),
                "trace_ep": coordinates.get("trace_ep"),
                "trace_sequence_length": coordinates.get("trace_sequence_length"),
                "treatment": treatment_name,
                "oracle_policy": rows[0]["oracle_policy"],
                "objective_unit": rows[0]["objective_unit"],
                "fixture_count": len(rows),
                "window_count": window_count,
                "objective_sum": objective_sum,
                "logical_fifo_reference_id": rows[0]["logical_fifo_reference_id"],
                "logical_fifo_objective_sum": fifo_sum,
                "relative_to_logical_fifo_ppm": relative_ppm,
                "relative_to_logical_fifo_percent": (
                    None if relative_ppm is None else relative_ppm / 10_000.0
                ),
                "all_windows_feasible": all(bool(row["all_windows_feasible"]) for row in rows),
                "all_windows_certified_optimal": all(
                    bool(row["all_windows_certified_optimal"]) for row in rows
                ),
                "certified_window_count": certified_count,
                "uncertified_window_count": window_count - certified_count,
                "certification_rate_ppm": (
                    int(round(certified_count * 1_000_000 / window_count))
                    if window_count else 0
                ),
                "certification_rate_percent": (
                    certified_count * 100.0 / window_count if window_count else 0.0
                ),
                "evidence_scope": (
                    "LOGICAL_OPTIMAL_REFERENCE_NOT_RUNTIME_NS"
                    if certified_count == window_count
                    else "LOGICAL_FEASIBLE_UPPER_BOUND_WITH_MIP_GAP_NOT_RUNTIME_NS"
                ),
            })
    return grouped


def _prediction_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": row["fixture_id"],
            "trace_mode": row.get("trace_mode"),
            "trace_ep": row.get("trace_ep"),
            "trace_model": row.get("trace_model"),
            "trace_sequence_length": row.get("trace_sequence_length"),
            "trace_dataset_id": row.get("trace_dataset_id"),
            "trace_split": row.get("trace_split"),
            "trace_capture_id": row.get("trace_capture_id"),
            "trace_source_kind": row.get("trace_source_kind"),
            "trace_provenance_digest": row.get("trace_provenance_digest"),
            "repeat_index": row["repeat_index"],
            "treatment": row["treatment"]["name"],
            "scope": row["treatment"]["scope"],
            "information": row["treatment"]["information"],
            "prediction_absolute_error_bytes": row["prediction_absolute_error_bytes"],
            "prediction_relative_absolute_error_percent_mean": round(row["prediction_relative_absolute_error_ppm_mean"] / 10_000.0, 6),
            "prediction_matrix_overlap_percent_mean": round(row["prediction_matrix_overlap_ppm_mean"] / 10_000.0, 6),
            "prediction_top_destination_accuracy_percent_mean": round(row["prediction_top_destination_accuracy_ppm_mean"] / 10_000.0, 6),
            "window_makespan_ns_sum": row["window_makespan_ns_sum"],
            "run_forward_makespan_ns": row["run_forward_makespan_ns"],
        }
        for row in results
        if row["treatment"]["scope"] == "WINDOW_JOINT"
    ]


def _overhead_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": row["fixture_id"],
            "trace_mode": row.get("trace_mode"),
            "trace_ep": row.get("trace_ep"),
            "trace_model": row.get("trace_model"),
            "trace_sequence_length": row.get("trace_sequence_length"),
            "trace_dataset_id": row.get("trace_dataset_id"),
            "trace_split": row.get("trace_split"),
            "trace_capture_id": row.get("trace_capture_id"),
            "trace_source_kind": row.get("trace_source_kind"),
            "trace_provenance_digest": row.get("trace_provenance_digest"),
            "treatment": row["treatment"]["name"],
            "overlap": row["treatment"]["overlap"],
            "prediction_hidden_ns": row["prediction_hidden_ns"],
            "prediction_exposed_ns": row["prediction_exposed_ns"],
            "control_hidden_ns": row["control_hidden_ns"],
            "control_exposed_ns": row["control_exposed_ns"],
            "binding_hidden_ns": row["binding_hidden_ns"],
            "binding_exposed_ns": row["binding_exposed_ns"],
            "window_makespan_ns_sum": row["window_makespan_ns_sum"],
        }
        for row in results
        if row["treatment"]["core"] == "rscf"
    ]


def _isolated_worker_config(config: dict[str, Any]) -> dict[str, Any]:
    # A worker executes the same scientific contract as its parent.  Keep the
    # complete Oracle budget in the transaction spec even for non-Oracle
    # treatments because the common runtime/result builder records it.
    return {
        "name": str(config["name"]),
        "simulation": dict(config["simulation"]),
        "oracle": dict(config["oracle"]),
        "save_raw_events": bool(config["save_raw_events"]),
        "save_task_timeline": bool(config["save_task_timeline"]),
        "save_plans": bool(config["save_plans"]),
    }


@dataclasses.dataclass
class IsolatedRunHandle:
    proc: subprocess.Popen[Any]
    spec_path: Path
    status_path: Path
    log_path: Path
    log_handle: Any
    started: float
    timeout_seconds: int
    trusted_fixture: dict[str, Any] | None
    finished: bool = False


def _launch_one_isolated(
    *,
    fixture_path: Path,
    fixture_index: int,
    treatment: dict[str, str],
    treatment_index: int,
    repeat_index: int,
    warmup: bool,
    config: dict[str, Any],
    run_dir: Path,
    trusted_fixture: dict[str, Any] | None = None,
) -> IsolatedRunHandle:
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "run_spec.json"
    status_path = run_dir / "worker_status.json"
    log_path = run_dir / "worker.log"
    # A rerun owns the transaction directory.  Remove stale terminal markers
    # before launch so the parent never consumes a previous attempt.
    status_path.unlink(missing_ok=True)
    spec = {
        "schema_version": "RS_SIM_EXPERIMENT_WORKER_SPEC",
        "fixture_path": str(fixture_path),
        "fixture_index": int(fixture_index),
        "treatment": dict(treatment),
        "treatment_index": int(treatment_index),
        "repeat_index": int(repeat_index),
        "warmup": bool(warmup),
        "config": _isolated_worker_config(config),
        "run_dir": str(run_dir),
        "trusted_fixture": None if trusted_fixture is None else dict(trusted_fixture),
    }
    spec_path.write_text(
        json.dumps(_freeze(spec), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = source_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("PYTHONHASHSEED", "0")
    command = [
        sys.executable,
        "-m",
        "rs_sim.app.experiment_worker",
        "--spec",
        str(spec_path),
        "--status",
        str(status_path),
    ]
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    return IsolatedRunHandle(
        proc=proc,
        spec_path=spec_path,
        status_path=status_path,
        log_path=log_path,
        log_handle=log_handle,
        started=time.monotonic(),
        timeout_seconds=int(config["execution"]["per_run_timeout_seconds"]),
        trusted_fixture=None if trusted_fixture is None else dict(trusted_fixture),
    )


def _terminate_isolated_handle(handle: IsolatedRunHandle) -> bool:
    forced = False
    try:
        if hasattr(signal, "SIGKILL") and os.name != "nt":
            os.killpg(handle.proc.pid, signal.SIGKILL)
        else:
            handle.proc.kill()
        forced = True
    except ProcessLookupError:
        pass
    return forced


def _kill_isolated_process(proc: subprocess.Popen[Any]) -> bool:
    forced = False
    try:
        if hasattr(signal, "SIGKILL"):
            os.kill(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        forced = True
    except ProcessLookupError:
        pass
    return forced


def _poll_one_isolated(handle: IsolatedRunHandle) -> dict[str, Any] | None:
    """Return a committed result, raise on terminal failure, or return None."""
    if handle.finished:
        raise RuntimeError("isolated run handle already finished")
    observed_status: dict[str, Any] = {}
    if handle.status_path.is_file():
        try:
            observed_status = json.loads(handle.status_path.read_text(encoding="utf-8"))
        except Exception:
            observed_status = {}
    status = str(observed_status.get("status", ""))
    elapsed = round(time.monotonic() - handle.started, 6)
    if status in {"PASS", "FAILED"}:
        # The status/result files are the authoritative transaction boundary.
        # Some numerical runtimes can remain alive after committing them, and
        # Popen.poll()/waitpid may block indefinitely in those environments.
        # Terminate the already-committed worker by PID without polling or
        # waiting; retain the Popen object so its destructor cannot call waitpid
        # before the parent exits.
        forced = _kill_isolated_process(handle.proc)
        _RETIRED_ISOLATED_PROCESSES.append(handle.proc)
        handle.log_handle.close()
        handle.finished = True
        if status != "PASS":
            raise RunProcessError(
                "isolated experiment worker failed",
                details={
                    "return_code": -997 if forced else 2,
                    "timed_out": False,
                    "elapsed_seconds": elapsed,
                    "worker_status": observed_status,
                    "worker_log": str(handle.log_path),
                    "run_spec": str(handle.spec_path),
                    "forced_exit_after_status": forced,
                },
            )
        result_path = Path(str(observed_status["result_path"]))
        row = json.loads(result_path.read_text(encoding="utf-8"))
        row["worker_execution_mode"] = "SUBPROCESS_ISOLATED"
        row["worker_return_code"] = -997 if forced else 0
        row["worker_elapsed_ns"] = int(round(elapsed * 1_000_000_000))
        row["worker_forced_exit_after_status"] = bool(forced)
        row["worker_log"] = str(handle.log_path)
        row["result_path"] = str(result_path)
        result_path.write_text(stable_json_dumps(_freeze(row)) + "\n", encoding="utf-8")
        return row
    if time.monotonic() - handle.started >= handle.timeout_seconds:
        forced = _terminate_isolated_handle(handle)
        _RETIRED_ISOLATED_PROCESSES.append(handle.proc)
        handle.log_handle.close()
        handle.finished = True
        raise RunProcessError(
            "isolated experiment worker timed out",
            details={
                "return_code": -998,
                "timed_out": True,
                "elapsed_seconds": elapsed,
                "worker_status": observed_status,
                "worker_log": str(handle.log_path),
                "run_spec": str(handle.spec_path),
                "forced_exit_after_status": forced,
            },
        )
    return None


def _run_one_isolated(
    *,
    fixture_path: Path,
    fixture_index: int,
    treatment: dict[str, str],
    treatment_index: int,
    repeat_index: int,
    warmup: bool,
    config: dict[str, Any],
    run_dir: Path,
    trusted_fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    handle = _launch_one_isolated(
        fixture_path=fixture_path,
        fixture_index=fixture_index,
        treatment=treatment,
        treatment_index=treatment_index,
        repeat_index=repeat_index,
        warmup=warmup,
        config=config,
        run_dir=run_dir,
        trusted_fixture=trusted_fixture,
    )
    while True:
        result = _poll_one_isolated(handle)
        if result is not None:
            return result
        time.sleep(0.05)


def _bundle_results(output_dir: Path, name: str) -> tuple[Path, Path]:
    files = [path for path in sorted(output_dir.rglob("*")) if path.is_file() and "bundles" not in path.relative_to(output_dir).parts]
    manifest = output_dir / "ARTIFACT_MANIFEST.sha256"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in files if path != manifest),
        encoding="utf-8",
    )
    if manifest not in files:
        files.append(manifest)
    bundle_dir = output_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    bundle = bundle_dir / f"{safe}_results.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(output_dir).as_posix())
    sha_path = bundle.with_suffix(bundle.suffix + ".sha256")
    sha_path.write_text(f"{_sha256(bundle)}  {bundle.name}\n", encoding="utf-8")
    return bundle, sha_path


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_experiment_config(config)
    output_dir = Path(normalized["output_dir"])
    if output_dir.exists() and normalized["overwrite"]:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    treatments = _expand_treatments(
        normalized["experiments"],
        default_release_mode=normalized["simulation"]["release_mode"],
    )
    claim_mode = normalized["comparison"]["claim_mode"]
    if claim_mode not in {"DIAGNOSTIC", "PAPER"}:
        raise ConfigError("comparison.claim_mode must be DIAGNOSTIC or PAPER")
    if claim_mode == "PAPER":
        _validate_paper_claim_treatments(treatments)
        try:
            require_paper_execution_semantics(
                p0_p1_compute_end_barrier=normalized["simulation"]["p0_p1_compute_end_barrier"],
                max_task_bytes=normalized["simulation"]["max_task_bytes"],
                alignment_bytes=normalized["simulation"]["alignment_bytes"],
            )
            for treatment in treatments:
                require_paper_treatment_release_semantics(
                    scope=treatment["scope"],
                    release_mode=treatment["release_mode"],
                    experiment_role=treatment["experiment_role"],
                )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    treatment_names = {item["name"] for item in treatments}
    baseline_name = normalized["comparison"]["baseline"]
    if not baseline_name:
        baseline_name = next(
            (item["name"] for item in treatments if item["core"] == "null"),
            treatments[0]["name"],
        )
    if baseline_name not in treatment_names:
        raise ConfigError(
            f"comparison.baseline={baseline_name!r} is not a treatment; available={sorted(treatment_names)}"
        )
    for reference_name in normalized["comparison"]["reference_baselines"]:
        if reference_name not in treatment_names:
            raise ConfigError(
                f"comparison.reference_baselines contains {reference_name!r}, available={sorted(treatment_names)}"
            )
    for overlap_mode, reference_name in normalized["comparison"]["baselines_by_overlap"].items():
        if reference_name not in treatment_names:
            raise ConfigError(
                f"comparison.baselines_by_overlap[{overlap_mode}]={reference_name!r} is not a treatment"
            )
    temporary_roots: list[tempfile.TemporaryDirectory] = []
    try:
        fixtures: list[Path] = []
        for path in normalized["trace_paths"]:
            fixtures.extend(_fixtures_from_path(path, temporary_roots))
        fixtures = sorted(set(path.resolve() for path in fixtures), key=str)
        if not fixtures:
            raise ConfigError("no fixtures resolved from trace paths")
        runs_dir = output_dir / "runs"
        warmup_count = normalized["repetitions"]["warmup"]
        measure_count = normalized["repetitions"]["measure"]
        if measure_count <= 0 or warmup_count < 0:
            raise ConfigError("repetitions.measure must be positive and warmup nonnegative")
        measured: list[dict[str, Any]] = []
        oracle_measured: list[dict[str, Any]] = []
        all_status: list[dict[str, Any]] = []
        stop = False
        for fixture_index, fixture_path in enumerate(fixtures):
            if stop:
                break
            for sequence_index in range(warmup_count + measure_count):
                if stop:
                    break
                warmup = sequence_index < warmup_count
                logical_repeat = sequence_index if warmup else sequence_index - warmup_count
                repeat_label = f"warmup{logical_repeat:03d}" if warmup else f"measure{logical_repeat:03d}"
                for treatment_index, treatment in enumerate(treatments):
                    safe_treatment = "".join(
                        ch if ch.isalnum() or ch in "-_." else "_"
                        for ch in treatment["name"]
                    )
                    run_dir = runs_dir / f"fixture{fixture_index:03d}" / repeat_label / safe_treatment
                    try:
                        if normalized["execution"]["mode"] == "SUBPROCESS_ISOLATED":
                            row = _run_one_isolated(
                                fixture_path=fixture_path,
                                fixture_index=fixture_index,
                                treatment=treatment,
                                treatment_index=treatment_index,
                                repeat_index=logical_repeat,
                                warmup=warmup,
                                config=normalized,
                                run_dir=run_dir,
                            )
                        else:
                            row = _run_one(
                                fixture_path=fixture_path,
                                fixture_index=fixture_index,
                                treatment=treatment,
                                treatment_index=treatment_index,
                                repeat_index=logical_repeat,
                                warmup=warmup,
                                config=normalized,
                                run_dir=run_dir,
                            )
                            row["worker_execution_mode"] = normalized["execution"]["mode"]
                        all_status.append(
                            {
                                "status": "PASS",
                                "result_path": row["result_path"],
                                "fixture_id": row["fixture_id"],
                                "treatment": treatment["name"],
                                "repeat_index": logical_repeat,
                                "warmup": warmup,
                                "worker_execution_mode": row.get("worker_execution_mode", "INPROCESS_DEBUG"),
                            }
                        )
                        if not warmup:
                            measured.append(row)
                    except Exception as exc:
                        details = dict(getattr(exc, "details", {}))
                        failure = {
                            "schema_version": "RS_SIM_EXPERIMENT_RUN_FAILURE",
                            "status": "FAILED",
                            "fixture_index": int(fixture_index),
                            "treatment_index": int(treatment_index),
                            "fixture_path": str(fixture_path),
                            "treatment": treatment,
                            "repeat_index": logical_repeat,
                            "warmup": warmup,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            **details,
                        }
                        run_dir.mkdir(parents=True, exist_ok=True)
                        failure_path = run_dir / "failure.json"
                        failure_path.write_text(
                            json.dumps(_freeze(failure), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        all_status.append({**failure, "failure_path": str(failure_path)})
                        if normalized["execution"]["fail_fast"]:
                            stop = True
                            break

        _validate_fate_joint_identity(measured)

        if normalized["raw_only"]:
            complete_rows = _complete_csv_rows(
                measured,
                oracle_measured,
                all_status,
                normalized=normalized,
            )
            complete_csv_path = output_dir / normalized["complete_csv_filename"]
            _write_complete_results_csv(complete_csv_path, complete_rows)
            status_counts: dict[str, int] = {}
            for status_row in all_status:
                status = str(status_row.get("status", "UNKNOWN"))
                status_counts[status] = status_counts.get(status, 0) + 1
            safe_bundle_name = "".join(
                ch if ch.isalnum() or ch in "-_." else "_" for ch in normalized["name"]
            ) + "_results.zip"
            manifest = {
                "schema_version": "RS_SIM_RAW_EXPERIMENT_RESULT_MANIFEST",
                "status": "PASS" if status_counts.get("FAILED", 0) == 0 else "PARTIAL",
                "name": normalized["name"],
                "reporting_mode": "RAW_ONLY_SINGLE_CSV",
                "trace_roots": [str(path) for path in normalized["trace_paths"]],
                "fixture_count": len(fixtures),
                "treatments": treatments,
                "warmup": warmup_count,
                "measure": measure_count,
                "execution": normalized["execution"],
                "status_counts": status_counts,
                "complete_results_csv_path": str(complete_csv_path),
                "complete_results_row_count": len(complete_rows),
                "run_status_path": str(output_dir / "run_status.json"),
                "result_bundle": f"bundles/{safe_bundle_name}",
                "result_bundle_sha256_file": f"bundles/{safe_bundle_name}.sha256",
            }
            (output_dir / "experiment_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "run_status.json").write_text(
                json.dumps(all_status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _bundle_results(output_dir, normalized["name"])
            return manifest

        summary = _summary_rows(measured)
        run_pairs, window_pairs = _paired_rows(
            measured,
            baseline_name=baseline_name,
            baselines_by_overlap=normalized["comparison"]["baselines_by_overlap"],
            reference_baselines=normalized["comparison"]["reference_baselines"],
        )
        pair_summary = _paired_summary_rows(
            run_pairs,
            window_pairs,
            target_improvement_percent=normalized["comparison"]["target_improvement_percent"],
            tie_tolerance_percent=normalized["comparison"]["tie_tolerance_percent"],
        )
        grouped_pair_summary = _grouped_paired_summary_rows(
            run_pairs,
            window_pairs,
            target_improvement_percent=normalized["comparison"]["target_improvement_percent"],
            tie_tolerance_percent=normalized["comparison"]["tie_tolerance_percent"],
        )
        runtime_fifo_relative = [
            row for row in grouped_pair_summary if row["baseline"] == baseline_name
        ]
        prediction = _prediction_rows(measured)
        overhead = _overhead_rows(measured)
        complete_rows = _complete_csv_rows(
            measured,
            oracle_measured,
            all_status,
            normalized=normalized,
        )
        complete_csv_path = output_dir / normalized["complete_csv_filename"]
        _write_complete_results_csv(complete_csv_path, complete_rows)
        _write_csv(output_dir / "summary.csv", summary, list(summary[0].keys()) if summary else ["fixture_id"])
        _write_csv(output_dir / "paired_comparison.csv", run_pairs, list(run_pairs[0].keys()) if run_pairs else ["fixture_id"])
        _write_csv(output_dir / "window_paired_comparison.csv", window_pairs, list(window_pairs[0].keys()) if window_pairs else ["fixture_id"])
        _write_csv(output_dir / "paired_summary.csv", pair_summary, list(pair_summary[0].keys()) if pair_summary else ["baseline"])
        _write_csv(
            output_dir / "paired_summary_grouped.csv",
            grouped_pair_summary,
            list(grouped_pair_summary[0].keys()) if grouped_pair_summary else ["group_kind"],
        )
        _write_csv(
            output_dir / "runtime_fifo_relative.csv",
            runtime_fifo_relative,
            list(runtime_fifo_relative[0].keys()) if runtime_fifo_relative else ["group_kind"],
        )
        _write_csv(output_dir / "prediction.csv", prediction, list(prediction[0].keys()) if prediction else ["fixture_id"])
        _write_csv(output_dir / "overhead.csv", overhead, list(overhead[0].keys()) if overhead else ["fixture_id"])

        oracle_window_rows: list[dict[str, Any]] = []
        oracle_summary_rows: list[dict[str, Any]] = []
        for oracle_row in oracle_measured:
            oracle_summary_rows.append({
                "fixture_id": oracle_row["fixture_id"],
                "trace_mode": oracle_row.get("trace_mode"),
                "trace_ep": oracle_row.get("trace_ep"),
                "trace_model": oracle_row.get("trace_model"),
                "trace_sequence_length": oracle_row.get("trace_sequence_length"),
                "trace_dataset_id": oracle_row.get("trace_dataset_id"),
                "trace_split": oracle_row.get("trace_split"),
                "trace_capture_id": oracle_row.get("trace_capture_id"),
                "trace_source_kind": oracle_row.get("trace_source_kind"),
                "trace_provenance_digest": oracle_row.get("trace_provenance_digest"),
                "treatment": oracle_row["treatment"]["name"],
                "oracle_policy": oracle_row["oracle_policy"],
                "objective_unit": oracle_row["objective_unit"],
                "window_count": oracle_row["window_count"],
                "objective_sum": oracle_row["objective_sum"],
                "logical_fifo_reference_id": oracle_row["logical_fifo_reference_id"],
                "logical_fifo_objective_sum": oracle_row["logical_fifo_objective_sum"],
                "relative_to_logical_fifo_ppm": oracle_row["relative_to_logical_fifo_ppm"],
                "relative_to_logical_fifo_percent": (
                    None
                    if oracle_row["relative_to_logical_fifo_ppm"] is None
                    else oracle_row["relative_to_logical_fifo_ppm"] / 10_000.0
                ),
                "require_all_certified": oracle_row["require_all_certified"],
                "all_windows_feasible": oracle_row["all_windows_feasible"],
                "all_windows_certified_optimal": oracle_row["all_windows_certified_optimal"],
                "certified_window_count": oracle_row["certified_window_count"],
                "uncertified_window_count": oracle_row["uncertified_window_count"],
                "certification_rate_ppm": oracle_row["certification_rate_ppm"],
                "certification_rate_percent": oracle_row["certification_rate_ppm"] / 10_000.0,
                "evidence_scope": oracle_row["evidence_scope"],
                "oracle_model_id": oracle_row["oracle_model_id"],
            })
            for window in oracle_row["per_window_oracle"]:
                oracle_window_rows.append({
                    "fixture_id": oracle_row["fixture_id"],
                    "trace_mode": oracle_row.get("trace_mode"),
                    "trace_ep": oracle_row.get("trace_ep"),
                    "trace_model": oracle_row.get("trace_model"),
                    "trace_sequence_length": oracle_row.get("trace_sequence_length"),
                    "trace_dataset_id": oracle_row.get("trace_dataset_id"),
                    "trace_split": oracle_row.get("trace_split"),
                    "trace_capture_id": oracle_row.get("trace_capture_id"),
                    "trace_source_kind": oracle_row.get("trace_source_kind"),
                    "trace_provenance_digest": oracle_row.get("trace_provenance_digest"),
                    "treatment": oracle_row["treatment"]["name"],
                    **window,
                    "relative_to_logical_fifo_percent": (
                        None
                        if window.get("relative_to_logical_fifo_ppm") is None
                        else window["relative_to_logical_fifo_ppm"] / 10_000.0
                    ),
                })
        oracle_grouped_rows = _grouped_oracle_reference_rows(oracle_measured)
        _write_csv(
            output_dir / "oracle_reference.csv",
            oracle_window_rows,
            list(oracle_window_rows[0].keys()) if oracle_window_rows else ["fixture_id"],
        )
        _write_csv(
            output_dir / "oracle_reference_summary.csv",
            oracle_summary_rows,
            list(oracle_summary_rows[0].keys()) if oracle_summary_rows else ["fixture_id"],
        )
        _write_csv(
            output_dir / "oracle_fifo_relative.csv",
            oracle_grouped_rows,
            list(oracle_grouped_rows[0].keys()) if oracle_grouped_rows else ["group_kind"],
        )
        (output_dir / "oracle_reference.json").write_text(
            json.dumps({
                "schema_version": "RS_SIM_ORACLE_REFERENCE_BUNDLE",
                "results": oracle_measured,
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        status_counts: dict[str, int] = {}
        for row in all_status:
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        fairness_pass = bool(run_pairs) and all(bool(row["fairness_pass"]) for row in run_pairs)
        minimum_samples = int(normalized["comparison"]["minimum_paired_samples"])
        sufficient_pairs = all(
            int(row["paired_window_count"]) >= minimum_samples for row in pair_summary
        ) if pair_summary else False
        synthetic_target_met = any(
            bool(row["synthetic_30_percent_target_met"]) for row in pair_summary
        )
        audit = {
            "schema_version": "RS_SIM_PAPER_EVIDENCE_AUDIT",
            "status": "MECHANISM_READY_PERFORMANCE_BLOCKED",
            "baseline": baseline_name,
            "baselines_by_overlap": normalized["comparison"]["baselines_by_overlap"],
            "reference_baselines": normalized["comparison"]["reference_baselines"],
            "target_improvement_percent": normalized["comparison"]["target_improvement_percent"],
            "fairness_pass": fairness_pass,
            "sufficient_paired_samples": sufficient_pairs,
            "synthetic_target_met_by_any_candidate": synthetic_target_met,
            "formal_runtime_correctness_pass": status_counts.get("FAILED", 0) == 0,
            "hardware_profile_calibrated": False,
            "performance_claim_allowed": False,
            "ttft_claim_allowed": False,
            "ttft_metric_scope": "MOE_FORWARD_ONLY_PROXY_NOT_SERVICE_TTFT",
            "paper_claim_blockers": [
                "hardware profile is not calibrated",
                "trace may be synthetic or capture marked performance_eligible=false",
                "compute-network interference is not calibrated",
                "TTFT proxy excludes tokenizer, dense-only path, KV-cache and serving queueing",
            ],
            "paired_summaries": pair_summary,
        }
        (output_dir / "paper_evidence_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics_payload = {
            "schema_version": "RS_SIM_EXPERIMENT_METRICS",
            "baseline": baseline_name,
            "summary": summary,
            "paired_summary": pair_summary,
            "paired_summary_grouped": grouped_pair_summary,
            "runtime_fifo_relative": runtime_fifo_relative,
            "prediction": prediction,
            "overhead": overhead,
            "oracle_reference_summary": oracle_summary_rows,
            "oracle_fifo_relative": oracle_grouped_rows,
        }
        (output_dir / "metrics_summary.json").write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        safe_bundle_name = "".join(
            ch if ch.isalnum() or ch in "-_." else "_" for ch in normalized["name"]
        ) + "_results.zip"
        manifest = {
            "schema_version": "RS_SIM_EXPERIMENT_RESULT_MANIFEST",
            "status": "PASS" if status_counts.get("FAILED", 0) == 0 else "PARTIAL",
            "name": normalized["name"],
            "trace_roots": [str(path) for path in normalized["trace_paths"]],
            "fixture_count": len(fixtures),
            "fixture_paths": [str(path) for path in fixtures],
            "treatments": treatments,
            "comparison_baseline": baseline_name,
            "warmup": warmup_count,
            "measure": measure_count,
            "execution": normalized["execution"],
            "oracle": normalized["oracle"],
            "status_counts": status_counts,
            "complete_results_csv_path": str(complete_csv_path),
            "complete_results_row_count": len(complete_rows),
            "summary_path": str(output_dir / "summary.csv"),
            "paired_comparison_path": str(output_dir / "paired_comparison.csv"),
            "window_paired_comparison_path": str(output_dir / "window_paired_comparison.csv"),
            "paired_summary_path": str(output_dir / "paired_summary.csv"),
            "paired_summary_grouped_path": str(output_dir / "paired_summary_grouped.csv"),
            "runtime_fifo_relative_path": str(output_dir / "runtime_fifo_relative.csv"),
            "prediction_path": str(output_dir / "prediction.csv"),
            "overhead_path": str(output_dir / "overhead.csv"),
            "oracle_reference_path": str(output_dir / "oracle_reference.csv"),
            "oracle_reference_summary_path": str(output_dir / "oracle_reference_summary.csv"),
            "oracle_fifo_relative_path": str(output_dir / "oracle_fifo_relative.csv"),
            "oracle_reference_json_path": str(output_dir / "oracle_reference.json"),
            "paper_evidence_audit_path": str(output_dir / "paper_evidence_audit.json"),
            "metrics_summary_path": str(output_dir / "metrics_summary.json"),
            "performance_claim_allowed": False,
            "result_bundle": f"bundles/{safe_bundle_name}",
            "result_bundle_sha256_file": f"bundles/{safe_bundle_name}.sha256",
        }
        (output_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "run_status.json").write_text(
            json.dumps(all_status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _bundle_results(output_dir, normalized["name"])
        return manifest
    finally:
        for temp in temporary_roots:
            temp.cleanup()
