from __future__ import annotations

"""Crash-resilient recursive trace-repository experiment sweep.

The CSV is the authoritative append journal.  Every completed treatment is
flushed and fsynced immediately; each fixture also emits a TRACE_COMPLETE row.
A restarted sweep derives stable run keys from trace truth, treatment semantics,
and settings, then skips already committed PASS rows.
"""

import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rs_sim.contracts.digest import stable_digest
from rs_sim.contracts.paper_defaults import (
    require_paper_execution_semantics,
    require_paper_treatment_release_semantics,
)
from rs_sim.trace.io.serialization import load_fixture
from rs_sim.trace.schema.validation import validate_fixture

from .config_io import ConfigError
from .experiment import (
    _complete_csv_row,
    _expand_treatments,
    _fixtures_from_path,
    _launch_one_isolated,
    _poll_one_isolated,
    _run_one,
    _run_one_isolated,
    _trace_identity_fields,
    _validate_paper_claim_treatments,
    normalize_experiment_config,
)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while limit > 131072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


@dataclass(frozen=True, slots=True)
class DiscoveredFixture:
    source_root: Path
    discovery_root: Path
    fixture_path: Path
    relative_path: str


_DESCRIPTION_KEYS = {
    "record_type", "status", "experiment_name", "config_path", "trace_roots",
    "fixture_index", "fixture_id", "fixture_path", "fixture_truth_digest",
    "trace_mode", "trace_model", "trace_ep", "trace_sequence_length",
    "trace_fixture_name", "trace_dataset_id", "trace_split", "trace_capture_id",
    "trace_collector_version", "trace_source_kind", "trace_notes",
    "trace_provenance_digest", "trace_source_digest", "trace_transform_digest",
    "world_size", "repeat_index", "warmup", "treatment_name", "paired_instance_id",
    "result_path", "failure_path", "worker_execution_mode",
}
_SETTING_KEYS = {
    "algorithm_core", "algorithm_policy", "scope", "planning", "information",
    "overlap", "experiment_role", "release_mode", "max_task_bytes",
    "max_window_prefix_tasks", "p0_p1_compute_end_barrier",
    "global_safe_selector_enabled", "hardware_profile_calibrated",
    "performance_claim_allowed", "oracle_policy", "objective_unit",
}
_METRIC_MARKERS = (
    "_mean", "_median", "_p50", "_p90", "_p95", "_p99", "_percent",
    "_ratio", "_spread", "_slowdown", "_improvement", "_sum", "_max",
    "communication_induced", "compute_excluded_communication", "ttft_proxy",
    "relative_gap", "optimality_gap", "best_bound", "objective_units",
)
_CATEGORY_ORDER = {"description": 0, "setting": 1, "observation": 2, "metric": 3}
_DESCRIPTION_PRIORITY = (
    "description__record_type", "description__status", "description__run_key",
    "description__trace_key", "description__experiment_name",
    "description__trace_root", "description__trace_relative_path",
    "description__trace_case_name", "description__run_name",
    "description__fixture_id", "description__fixture_truth_digest",
    "description__trace_model", "description__trace_ep",
    "description__trace_sequence_length", "description__treatment_name",
)


def _category_for_key(key: str) -> str:
    lowered = key.lower()
    if key in _DESCRIPTION_KEYS or lowered.startswith("trace_") or lowered.endswith("_path"):
        return "description"
    if key in _SETTING_KEYS or lowered.startswith("config__") or lowered.startswith("treatment__"):
        return "setting"
    if any(marker in lowered for marker in _METRIC_MARKERS):
        return "metric"
    return "observation"


def _field_sort_key(field: str) -> tuple[int, int, str]:
    if field in _DESCRIPTION_PRIORITY:
        return (0, _DESCRIPTION_PRIORITY.index(field), field)
    category = field.split("__", 1)[0]
    return (_CATEGORY_ORDER.get(category, 9), 1000, field)




def _digest_any(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _safe_component(value: Any, *, limit: int = 96) -> str:
    text = str(value or "unknown").strip().replace("\\", "/")
    text = "__".join(part for part in text.split("/") if part not in {"", "."})
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    safe = safe.strip("_.") or "unknown"
    if len(safe) <= limit:
        return safe
    suffix = stable_digest(safe)[:12]
    return f"{safe[:limit-14]}__{suffix}"


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DurableWideCsvJournal:
    """Append-only wide CSV with deterministic schema expansion."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields: list[str] = []
        if self.path.is_file() and self.path.stat().st_size:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                self.fields = next(reader, [])

    def committed_status_by_run_key(self) -> dict[str, str]:
        if not self.path.is_file() or not self.fields:
            return {}
        result: dict[str, str] = {}
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = str(row.get("description__run_key", "")).strip()
                if key:
                    result[key] = str(row.get("description__status", "")).upper()
        return result

    def committed_runtime_results(self, trace_key: str) -> dict[str, dict[str, Any]]:
        """Load durable PASS result rows already committed for one trace.

        This makes an interrupted/resumed fixture scientifically complete: an
        Oracle validation after restart sees both the old and newly completed
        treatments instead of only the latter subset.
        """
        if not self.path.is_file() or not self.fields:
            return {}
        result: dict[str, dict[str, Any]] = {}
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("description__record_type", "")) != "RUNTIME":
                    continue
                if str(row.get("description__status", "")).upper() != "PASS":
                    continue
                if str(row.get("description__trace_key", "")) != str(trace_key):
                    continue
                run_key = str(row.get("description__run_key", "")).strip()
                result_path = str(row.get("description__result_path", "")).strip()
                if not run_key or not result_path:
                    continue
                path = Path(result_path)
                if not path.is_file():
                    continue
                try:
                    result[run_key] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return result

    def _expand_schema(self, required: Iterable[str]) -> None:
        target = sorted(set(self.fields).union(required), key=_field_sort_key)
        if target == self.fields:
            return
        if not self.path.is_file() or self.path.stat().st_size == 0:
            self.fields = target
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self.fields).writeheader()
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.path.parent)
            return

        temporary = self.path.with_name(self.path.name + ".rewrite.tmp")
        with self.path.open("r", encoding="utf-8", newline="") as source, temporary.open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=target, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                writer.writerow(row)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, self.path)
        _fsync_directory(self.path.parent)
        self.fields = target

    def append(self, row: dict[str, Any]) -> None:
        scalar = {
            str(key): (
                value
                if value is None or isinstance(value, (str, int, float, bool))
                else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            for key, value in row.items()
        }
        self._expand_schema(scalar.keys())
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="ignore")
            writer.writerow(scalar)
            handle.flush()
            os.fsync(handle.fileno())


def _discover_fixtures(
    roots: Iterable[Path],
    temporary_roots: list[tempfile.TemporaryDirectory],
) -> tuple[DiscoveredFixture, ...]:
    discovered: list[DiscoveredFixture] = []
    seen: set[Path] = set()
    for source_root in roots:
        source_root = source_root.expanduser().resolve()
        before = len(temporary_roots)
        fixtures = _fixtures_from_path(source_root, temporary_roots)
        if source_root.is_dir():
            discovery_root = source_root
        elif len(temporary_roots) > before:
            discovery_root = Path(temporary_roots[-1].name).resolve()
        else:
            discovery_root = source_root.parent
        for fixture_path in fixtures:
            fixture_path = fixture_path.resolve()
            if fixture_path in seen:
                continue
            seen.add(fixture_path)
            try:
                relative = fixture_path.relative_to(discovery_root).as_posix()
            except ValueError:
                relative = fixture_path.name
            discovered.append(
                DiscoveredFixture(source_root, discovery_root, fixture_path, relative)
            )
    return tuple(sorted(discovered, key=lambda item: (str(item.source_root), item.relative_path)))


def _classify_discovered_fixture(item: DiscoveredFixture) -> str:
    """Classify trace evidence without reloading a fixture.

    Unified/projected repositories carry an explicit ``projected`` path
    component.  Everything else is treated as measured so standalone measured
    archives remain backward compatible.
    """

    components = [part.lower() for part in item.fixture_path.parts]
    source_components = [part.lower() for part in item.source_root.parts]
    if "projected" in components or "projected" in source_components:
        return "projected"
    return "measured"


def _filter_discovered_fixtures(
    fixtures: tuple[DiscoveredFixture, ...], *, trace_kind: str
) -> tuple[DiscoveredFixture, ...]:
    normalized = str(trace_kind).strip().lower()
    if normalized not in {"measured", "projected", "all"}:
        raise ConfigError("trace_kind must be measured, projected, or all")
    if normalized == "all":
        return fixtures
    selected = tuple(
        item for item in fixtures if _classify_discovered_fixture(item) == normalized
    )
    if not selected:
        raise ConfigError(f"no {normalized} trace fixtures found under the supplied roots")
    return selected


def _validate_contract(normalized: dict[str, Any], treatments: tuple[dict[str, str], ...]) -> str:
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
    names = {item["name"] for item in treatments}
    baseline = normalized["comparison"]["baseline"] or next(
        (item["name"] for item in treatments if item["core"] == "null"),
        treatments[0]["name"],
    )
    if baseline not in names:
        raise ConfigError(f"comparison baseline {baseline!r} is not a treatment")
    return baseline


def _categorize_row(
    row: dict[str, Any],
    *,
    normalized: dict[str, Any],
    record_type: str,
    run_key: str,
    trace_key: str,
    source_root: Path,
    relative_path: str,
    case_name: str,
    run_name: str,
) -> dict[str, Any]:
    flat = _complete_csv_row(row, record_type=record_type, normalized=normalized)
    result: dict[str, Any] = {
        "description__record_type": record_type,
        "description__status": str(row.get("status", "PASS")),
        "description__run_key": run_key,
        "description__trace_key": trace_key,
        "description__trace_root": str(source_root),
        "description__trace_relative_path": relative_path,
        "description__trace_case_name": case_name,
        "description__run_name": run_name,
    }
    for key, value in flat.items():
        result.setdefault(f"{_category_for_key(key)}__{key}", value)
    return result


def _row_field(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row.get(key)
    treatment = row.get("treatment")
    if isinstance(treatment, dict):
        aliases = {
            "algorithm_core": "core",
            "treatment_name": "name",
        }
        return treatment.get(aliases.get(key, key), default)
    return default


def _row_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _oracle_validation_records(
    rows: Iterable[dict[str, Any]],
    *,
    primary_metric: str,
) -> tuple[dict[str, Any], ...]:
    """Validate Oracle references against completed full-runtime treatments.

    This is deliberately a result-level guard.  Solver certification and the
    lightweight ready-aware replay remain separate diagnostics; paper-facing
    Oracle references must also dominate the relevant completed Backend runs.
    """

    items = tuple(row for row in rows if not bool(row.get("warmup", False)))
    by_repeat: dict[int, list[dict[str, Any]]] = {}
    for row in items:
        by_repeat.setdefault(int(row.get("repeat_index", 0)), []).append(row)

    records: list[dict[str, Any]] = []
    for repeat_index, repeat_rows in sorted(by_repeat.items()):
        local_oracles = [
            row for row in repeat_rows
            if str(_row_field(row, "algorithm_core", "")).lower() == "oracle"
            and str(_row_field(row, "scope", "")).upper() == "PHASE_LOCAL"
        ]
        joint_oracles = [
            row for row in repeat_rows
            if str(_row_field(row, "algorithm_core", "")).lower() == "oracle"
            and str(_row_field(row, "scope", "")).upper() == "WINDOW_JOINT"
        ]
        local_refs = [
            row for row in repeat_rows
            if str(_row_field(row, "scope", "")).upper() == "PHASE_LOCAL"
            and str(_row_field(row, "algorithm_core", "")).lower() in {"fifo", "birkhoff", "rscf"}
        ]
        perfect_joint_refs = [
            row for row in repeat_rows
            if str(_row_field(row, "scope", "")).upper() == "WINDOW_JOINT"
            and str(_row_field(row, "algorithm_core", "")).lower() == "rscf"
            and str(_row_field(row, "information", "")).upper() in {"PERFECT", "PERFECT_P2"}
        ]

        def metric_rows(candidates: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
            result: list[tuple[dict[str, Any], float]] = []
            for candidate in candidates:
                value = _row_metric(candidate, primary_metric)
                if value is not None:
                    result.append((candidate, value))
            return result

        for kind, oracle_rows in (("LOCAL", local_oracles), ("JOINT", joint_oracles)):
            candidates = metric_rows(oracle_rows)
            if not candidates:
                continue
            oracle_row, oracle_value = min(candidates, key=lambda item: item[1])
            comparator_pairs: list[tuple[str, float]] = []
            missing: list[str] = []
            if kind == "LOCAL":
                refs = metric_rows(local_refs)
                if refs:
                    comparator_pairs.extend(
                        (str(_row_field(row, "treatment_name", _row_field(row, "algorithm_core", "local_ref"))), value)
                        for row, value in refs
                    )
                else:
                    missing.append("FIFO/BIRKHOFF/RSCF_LOCAL")
            else:
                local_candidates = metric_rows(local_oracles)
                if local_candidates:
                    row, value = min(local_candidates, key=lambda item: item[1])
                    comparator_pairs.append((str(_row_field(row, "treatment_name", "Oracle-Local")), value))
                else:
                    missing.append("ORACLE_LOCAL")
                perfect_candidates = metric_rows(perfect_joint_refs)
                if perfect_candidates:
                    row, value = min(perfect_candidates, key=lambda item: item[1])
                    comparator_pairs.append((str(_row_field(row, "treatment_name", "RSCF-Joint-Perfect")), value))
                else:
                    missing.append("RSCF_JOINT_PERFECT")

            violations = tuple(
                (name, value)
                for name, value in comparator_pairs
                if oracle_value > value
            )
            if missing:
                status = "INCOMPLETE"
            elif violations:
                status = "INVALID_ORACLE_REFERENCE"
            else:
                status = "PASS"
            records.append({
                "status": status,
                "repeat_index": repeat_index,
                "oracle_validation_kind": kind,
                "oracle_treatment_name": _row_field(oracle_row, "treatment_name"),
                "oracle_primary_metric": primary_metric,
                "oracle_primary_metric_value": oracle_value,
                "oracle_reference_valid": status == "PASS",
                "oracle_paper_eligible": status == "PASS",
                "oracle_missing_comparators": tuple(missing),
                "oracle_dominance_violations": tuple(violations),
                "oracle_comparator_values": tuple(comparator_pairs),
            })
    return tuple(records)


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def run_repository_sweep(
    config: dict[str, Any],
    *,
    trace_roots: Iterable[str | Path],
    output_csv: str | Path,
    resume: bool = True,
    rerun_failures: bool = True,
    max_fixtures: int | None = None,
    workers: int = 1,
    trace_kind: str = "all",
) -> dict[str, Any]:
    roots = tuple(Path(value).expanduser().resolve() for value in trace_roots)
    if not roots:
        raise ConfigError("at least one --trace-root is required")
    csv_path = Path(output_csv).expanduser().resolve()

    source = dict(config)
    source.pop("trace", None)
    source["traces"] = [str(path) for path in roots]
    output = dict(source.get("output") or {})
    output["directory"] = str(csv_path.parent)
    output["complete_csv_filename"] = csv_path.name
    output["overwrite"] = False
    output["raw_only"] = True
    source["output"] = output
    normalized = normalize_experiment_config(source)
    treatments = _expand_treatments(
        normalized["experiments"],
        default_release_mode=normalized["simulation"]["release_mode"],
    )
    baseline_name = _validate_contract(normalized, treatments)
    worker_count = int(workers)
    if worker_count <= 0:
        raise ConfigError("workers must be positive")
    if worker_count > 1 and normalized["execution"]["mode"] != "SUBPROCESS_ISOLATED":
        raise ConfigError("workers > 1 requires execution.mode=SUBPROCESS_ISOLATED")

    if csv_path.exists() and not resume:
        raise ConfigError(f"output CSV already exists; use --resume or choose another path: {csv_path}")
    journal = DurableWideCsvJournal(csv_path)
    committed = journal.committed_status_by_run_key() if resume else {}
    temporary_roots: list[tempfile.TemporaryDirectory] = []
    progress_path = csv_path.with_suffix(csv_path.suffix + ".progress.json")
    manifest_path = csv_path.with_suffix(csv_path.suffix + ".manifest.json")
    runs_dir = csv_path.parent / f"{csv_path.stem}_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed_runs = 0
    skipped_runs = 0
    failed_runs = 0
    unresolved_failures = 0
    processed_fixtures = 0
    skipped_fixtures = 0
    invalid_oracle_references = 0
    incomplete_oracle_validations = 0

    try:
        fixtures = list(
            _filter_discovered_fixtures(
                _discover_fixtures(roots, temporary_roots), trace_kind=trace_kind
            )
        )
        if max_fixtures is not None:
            fixtures = fixtures[: max(0, int(max_fixtures))]
        if not fixtures:
            raise ConfigError("no fixtures resolved recursively from trace roots")

        config_semantics = _digest_any({
            "simulation": normalized["simulation"],
            "repetitions": normalized["repetitions"],
            "execution": normalized["execution"],
            "oracle": normalized["oracle"],
            "comparison": normalized["comparison"],
        })

        for fixture_index, discovered in enumerate(fixtures):
            fixture_started = time.monotonic()
            fixture_rows: list[dict[str, Any]] = []
            try:
                fixture = load_fixture(discovered.fixture_path)
                validation = validate_fixture(fixture)
                if str(validation.get("status", "")).upper() != "PASS":
                    raise ConfigError(f"fixture validation failed: {validation}")
                identity = _trace_identity_fields(discovered.fixture_path, fixture)
                truth_digest = fixture.truth_digest()
                fixture_stat = discovered.fixture_path.stat()
                trusted_fixture = {
                    "truth_digest": truth_digest,
                    "size_bytes": int(fixture_stat.st_size),
                    "mtime_ns": int(fixture_stat.st_mtime_ns),
                }
                trace_key = stable_digest({
                    "source_root": str(discovered.source_root),
                    "relative_path": discovered.relative_path,
                    "fixture_truth_digest": truth_digest,
                })
                path_name = _safe_component(Path(discovered.relative_path).with_suffix("").as_posix())
                model_name = _safe_component(identity.get("trace_model") or "model", limit=48)
                ep_name = identity.get("trace_ep") or fixture.world_size
                seq_name = identity.get("trace_sequence_length") or "na"
                case_name = _safe_component(
                    f"{path_name}__{model_name}__ep{ep_name}__seq{seq_name}__{fixture.fixture_id}",
                    limit=180,
                )
            except Exception as exc:
                relative = discovered.relative_path
                trace_key = stable_digest({"source_root": str(discovered.source_root), "relative_path": relative})
                run_key = stable_digest({"trace_key": trace_key, "record": "TRACE_VALIDATION"})
                failure = {
                    "status": "FAILED",
                    "fixture_index": fixture_index,
                    "fixture_path": str(discovered.fixture_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                journal.append(_categorize_row(
                    failure,
                    normalized=normalized,
                    record_type="TRACE_FAILURE",
                    run_key=run_key,
                    trace_key=trace_key,
                    source_root=discovered.source_root,
                    relative_path=relative,
                    case_name=_safe_component(relative),
                    run_name="trace_validation",
                ))
                failed_runs += 1
                unresolved_failures += 1
                if normalized["execution"]["fail_fast"]:
                    break
                continue

            trace_complete_key = stable_digest({
                "trace_key": trace_key, "record": "TRACE_COMPLETE", "config": config_semantics
            })
            if resume and committed.get(trace_complete_key) == "PASS":
                skipped_fixtures += 1
                continue

            fixture_failed = 0
            fixture_rows_by_run_key = journal.committed_runtime_results(trace_key) if resume else {}
            total_sequences = normalized["repetitions"]["warmup"] + normalized["repetitions"]["measure"]

            def execute_treatment(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, BaseException | None]:
                try:
                    if normalized["execution"]["mode"] == "SUBPROCESS_ISOLATED":
                        row = _run_one_isolated(
                            fixture_path=discovered.fixture_path,
                            fixture_index=fixture_index,
                            treatment=spec["treatment"],
                            treatment_index=spec["treatment_index"],
                            repeat_index=spec["logical_repeat"],
                            warmup=spec["warmup"],
                            config=normalized,
                            run_dir=spec["run_dir"],
                            trusted_fixture=trusted_fixture,
                        )
                    else:
                        row = _run_one(
                            fixture_path=discovered.fixture_path,
                            fixture_index=fixture_index,
                            treatment=spec["treatment"],
                            treatment_index=spec["treatment_index"],
                            repeat_index=spec["logical_repeat"],
                            warmup=spec["warmup"],
                            config=normalized,
                            run_dir=spec["run_dir"],
                            trusted_fixture=trusted_fixture,
                        )
                        row["worker_execution_mode"] = normalized["execution"]["mode"]
                    return spec, row, None
                except BaseException as exc:
                    return spec, None, exc

            for sequence_index in range(total_sequences):
                warmup = sequence_index < normalized["repetitions"]["warmup"]
                logical_repeat = sequence_index if warmup else sequence_index - normalized["repetitions"]["warmup"]
                repeat_label = f"warmup{logical_repeat:03d}" if warmup else f"measure{logical_repeat:03d}"
                pending: list[dict[str, Any]] = []
                for treatment_index, treatment in enumerate(treatments):
                    run_key = stable_digest({
                        "trace_key": trace_key,
                        "fixture_truth_digest": truth_digest,
                        "treatment": treatment,
                        "repeat_index": logical_repeat,
                        "warmup": warmup,
                        "config_semantics": config_semantics,
                    })
                    previous = committed.get(run_key)
                    if resume and previous == "PASS":
                        skipped_runs += 1
                        continue
                    if resume and previous == "FAILED" and not rerun_failures:
                        skipped_runs += 1
                        fixture_failed += 1
                        unresolved_failures += 1
                        continue
                    treatment_name = _safe_component(treatment["name"], limit=72)
                    run_name = f"{case_name}__{treatment_name}__{repeat_label}"
                    pending.append({
                        "run_key": run_key,
                        "treatment": treatment,
                        "treatment_index": treatment_index,
                        "logical_repeat": logical_repeat,
                        "warmup": warmup,
                        "run_name": run_name,
                        "run_dir": runs_dir / case_name / repeat_label / treatment_name,
                    })

                if worker_count == 1:
                    completed = (execute_treatment(spec) for spec in pending)
                else:
                    def parallel_completed():
                        queue = list(pending)
                        active: list[tuple[Any, dict[str, Any]]] = []
                        while queue or active:
                            while queue and len(active) < worker_count:
                                spec = queue.pop(0)
                                try:
                                    handle = _launch_one_isolated(
                                        fixture_path=discovered.fixture_path,
                                        fixture_index=fixture_index,
                                        treatment=spec["treatment"],
                                        treatment_index=spec["treatment_index"],
                                        repeat_index=spec["logical_repeat"],
                                        warmup=spec["warmup"],
                                        config=normalized,
                                        run_dir=spec["run_dir"],
                                        trusted_fixture=trusted_fixture,
                                    )
                                    active.append((handle, spec))
                                except BaseException as exc:
                                    yield spec, None, exc
                            made_progress = False
                            for handle, spec in tuple(active):
                                try:
                                    row = _poll_one_isolated(handle)
                                except BaseException as exc:
                                    active.remove((handle, spec))
                                    made_progress = True
                                    yield spec, None, exc
                                    continue
                                if row is not None:
                                    active.remove((handle, spec))
                                    made_progress = True
                                    yield spec, row, None
                            if active and not made_progress:
                                time.sleep(0.05)
                    completed = parallel_completed()

                for spec, row, exc in completed:
                        run_key = str(spec["run_key"])
                        treatment = dict(spec["treatment"])
                        run_name = str(spec["run_name"])
                        if exc is None and row is not None:
                            row["status"] = "PASS"
                            if not bool(spec["warmup"]):
                                fixture_rows_by_run_key[run_key] = row
                            journal.append(_categorize_row(
                                row, normalized=normalized, record_type="RUNTIME",
                                run_key=run_key, trace_key=trace_key,
                                source_root=discovered.source_root,
                                relative_path=discovered.relative_path,
                                case_name=case_name, run_name=run_name,
                            ))
                            committed[run_key] = "PASS"
                            completed_runs += 1
                        else:
                            assert exc is not None
                            details = dict(getattr(exc, "details", {}))
                            failure = {
                                "status": "FAILED",
                                "fixture_index": fixture_index,
                                "fixture_id": fixture.fixture_id,
                                "fixture_path": str(discovered.fixture_path),
                                "fixture_truth_digest": truth_digest,
                                "treatment": treatment,
                                "repeat_index": int(spec["logical_repeat"]),
                                "warmup": bool(spec["warmup"]),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                **details,
                            }
                            journal.append(_categorize_row(
                                failure, normalized=normalized, record_type="FAILURE",
                                run_key=run_key, trace_key=trace_key,
                                source_root=discovered.source_root,
                                relative_path=discovered.relative_path,
                                case_name=case_name, run_name=run_name,
                            ))
                            committed[run_key] = "FAILED"
                            failed_runs += 1
                            unresolved_failures += 1
                            fixture_failed += 1
                            if normalized["execution"]["fail_fast"]:
                                raise exc

                        _write_progress(progress_path, {
                            "status": "RUNNING",
                            "output_csv": str(csv_path),
                            "fixture_count": len(fixtures),
                            "current_fixture_index": fixture_index,
                            "current_trace_relative_path": discovered.relative_path,
                            "last_treatment_name": treatment.get("name"),
                            "workers": worker_count,
                            "processed_fixtures": processed_fixtures,
                            "skipped_fixtures": skipped_fixtures,
                            "completed_runs": completed_runs,
                            "skipped_runs": skipped_runs,
                            "failed_runs": failed_runs,
                            "invalid_oracle_references": invalid_oracle_references,
                            "incomplete_oracle_validations": incomplete_oracle_validations,
                            "elapsed_seconds": round(time.monotonic() - started, 6),
                        })

            fixture_rows = list(fixture_rows_by_run_key.values())

            oracle_records = _oracle_validation_records(
                fixture_rows,
                primary_metric=normalized["comparison"]["primary_metric"],
            )
            fixture_oracle_invalid = 0
            fixture_oracle_incomplete = 0
            for validation_index, validation_row in enumerate(oracle_records):
                validation_key = stable_digest({
                    "trace_key": trace_key,
                    "record": "ORACLE_VALIDATION",
                    "repeat_index": validation_row.get("repeat_index"),
                    "kind": validation_row.get("oracle_validation_kind"),
                    "primary_metric": normalized["comparison"]["primary_metric"],
                    "config": config_semantics,
                })
                status_value = str(validation_row.get("status", "INCOMPLETE"))
                if status_value == "INVALID_ORACLE_REFERENCE":
                    fixture_oracle_invalid += 1
                    invalid_oracle_references += 1
                elif status_value == "INCOMPLETE":
                    fixture_oracle_incomplete += 1
                    incomplete_oracle_validations += 1
                journal.append(_categorize_row(
                    validation_row,
                    normalized=normalized,
                    record_type="ORACLE_VALIDATION",
                    run_key=validation_key,
                    trace_key=trace_key,
                    source_root=discovered.source_root,
                    relative_path=discovered.relative_path,
                    case_name=case_name,
                    run_name=f"{case_name}__oracle_validation_{validation_index:02d}",
                ))
                committed[validation_key] = status_value

            if fixture_failed:
                trace_status = "PARTIAL"
            elif fixture_oracle_invalid:
                trace_status = "INVALID_ORACLE_REFERENCE"
            else:
                trace_status = "PASS"
            journal.append({
                "description__record_type": "TRACE_COMPLETE",
                "description__status": trace_status,
                "description__run_key": trace_complete_key,
                "description__trace_key": trace_key,
                "description__trace_root": str(discovered.source_root),
                "description__trace_relative_path": discovered.relative_path,
                "description__trace_case_name": case_name,
                "description__run_name": f"{case_name}__complete",
                "description__fixture_id": fixture.fixture_id,
                "description__fixture_truth_digest": truth_digest,
                "observation__fixture_elapsed_seconds": round(time.monotonic() - fixture_started, 6),
                "observation__runtime_row_count": len(fixture_rows),
                "observation__failure_count": fixture_failed,
                "observation__oracle_invalid_count": fixture_oracle_invalid,
                "observation__oracle_incomplete_count": fixture_oracle_incomplete,
            })
            committed[trace_complete_key] = trace_status
            processed_fixtures += 1
            _write_progress(progress_path, {
                "status": "RUNNING",
                "output_csv": str(csv_path),
                "fixture_count": len(fixtures),
                "processed_fixtures": processed_fixtures,
                "skipped_fixtures": skipped_fixtures,
                "completed_runs": completed_runs,
                "skipped_runs": skipped_runs,
                "failed_runs": failed_runs,
                "invalid_oracle_references": invalid_oracle_references,
                "incomplete_oracle_validations": incomplete_oracle_validations,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "last_trace_relative_path": discovered.relative_path,
            })

        if unresolved_failures:
            status = "PARTIAL"
        elif invalid_oracle_references:
            status = "INVALID_ORACLE_REFERENCE"
        else:
            status = "PASS"
        manifest = {
            "schema_version": "RS_SIM_RECURSIVE_REPOSITORY_SWEEP",
            "status": status,
            "trace_roots": [str(path) for path in roots],
            "trace_kind": str(trace_kind).lower(),
            "output_csv": str(csv_path),
            "progress_path": str(progress_path),
            "fixture_count": len(fixtures),
            "processed_fixtures": processed_fixtures,
            "skipped_fixtures": skipped_fixtures,
            "treatment_count": len(treatments),
            "workers": worker_count,
            "completed_runs": completed_runs,
            "skipped_runs": skipped_runs,
            "failed_runs": failed_runs,
            "unresolved_failures": unresolved_failures,
            "invalid_oracle_references": invalid_oracle_references,
            "incomplete_oracle_validations": incomplete_oracle_validations,
            "resume": bool(resume),
            "rerun_failures": bool(rerun_failures),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
        _write_progress(manifest_path, manifest)
        _write_progress(progress_path, {**manifest, "status": status})
        return manifest
    finally:
        for temporary in temporary_roots:
            temporary.cleanup()


__all__ = [
    "DiscoveredFixture",
    "DurableWideCsvJournal",
    "run_repository_sweep",
]
