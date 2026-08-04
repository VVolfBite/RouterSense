from __future__ import annotations

import csv
from pathlib import Path

from rs_sim.app.formal_sweep import DurableWideCsvJournal, run_repository_sweep
from rs_sim.trace.schema.fixtures import write_builtin_fixtures


def test_durable_csv_expands_schema_and_recovers_run_status(tmp_path: Path) -> None:
    path = tmp_path / "journal.csv"
    journal = DurableWideCsvJournal(path)
    journal.append({
        "description__run_key": "a",
        "description__status": "PASS",
        "observation__value": 1,
    })
    journal.append({
        "description__run_key": "b",
        "description__status": "FAILED",
        "metric__p95": 2,
    })
    recovered = DurableWideCsvJournal(path)
    assert recovered.committed_status_by_run_key() == {"a": "PASS", "b": "FAILED"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "observation__value" in rows[0]
    assert "metric__p95" in rows[0]


def test_recursive_sweep_flushes_and_resumes_one_fixture(tmp_path: Path) -> None:
    trace_root = tmp_path / "repository" / "fixtures"
    paths = write_builtin_fixtures(trace_root)
    output_csv = tmp_path / "results" / "formal.csv"
    config = {
        "version": 1,
        "name": "test_recursive_sweep",
        "traces": [str(trace_root.parent)],
        "simulation": {
            "release_mode": "RANK_LOCAL",
            "p0_p1_compute_end_barrier": True,
            "staging": "1.0X",
            "max_task_bytes": 262144,
            "max_window_prefix_tasks": 64,
            "alignment_bytes": 256,
            "max_timestamps": 100000,
        },
        "experiments": {
            "treatments": [{
                "name": "FIFO-Local",
                "algorithm": "local(global_(fifo()))",
                "information": "zero",
                "overlap": "overlap",
                "experiment_role": "PAPER_BASELINE",
                "release_mode": "PHASE_BARRIER",
            }],
        },
        "repetitions": {"warmup": 0, "measure": 1, "seed": 1},
        "execution": {
            "mode": "INPROCESS_DEBUG",
            "per_run_timeout_seconds": 30,
            "kill_grace_seconds": 2,
            "fail_fast": True,
        },
        "oracle": {
            "time_limit_ms_per_window": 100,
            "relative_gap": 0.1,
            "require_all_certified": False,
        },
        "comparison": {
            "claim_mode": "DIAGNOSTIC",
            "baseline": "FIFO-Local",
            "primary_metric": "compute_excluded_communication_makespan_ns_sum",
            "target_improvement_percent": 0.0,
            "tie_tolerance_percent": 0.05,
            "minimum_paired_samples": 1,
        },
        "output": {
            "directory": str(output_csv.parent),
            "complete_csv_filename": output_csv.name,
            "save_raw_events": False,
            "save_task_timeline": False,
            "save_plans": False,
            "raw_only": True,
            "overwrite": False,
        },
        "__config_path": str(tmp_path / "config.json"),
        "__config_dir": str(tmp_path),
    }
    first = run_repository_sweep(
        config,
        trace_roots=[trace_root.parent],
        output_csv=output_csv,
        max_fixtures=1,
    )
    assert first["status"] == "PASS"
    assert first["completed_runs"] == 1
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows_before = list(csv.DictReader(handle))
    assert {row["description__record_type"] for row in rows_before} == {"RUNTIME", "TRACE_COMPLETE"}
    runtime = next(row for row in rows_before if row["description__record_type"] == "RUNTIME")
    assert any(runtime["description__trace_relative_path"].endswith(path.name) for path in paths)
    assert runtime["description__trace_case_name"]
    assert runtime["setting__algorithm_core"] == "fifo"

    second = run_repository_sweep(
        config,
        trace_roots=[trace_root.parent],
        output_csv=output_csv,
        resume=True,
        max_fixtures=1,
    )
    assert second["skipped_fixtures"] == 1
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows_after = list(csv.DictReader(handle))
    assert len(rows_after) == len(rows_before)


def test_oracle_validation_marks_runtime_dominance_violation() -> None:
    from rs_sim.app.formal_sweep import _oracle_validation_records

    rows = (
        {"repeat_index": 0, "algorithm_core": "fifo", "scope": "PHASE_LOCAL", "treatment_name": "FIFO-Local", "metric": 100},
        {"repeat_index": 0, "algorithm_core": "birkhoff", "scope": "PHASE_LOCAL", "treatment_name": "BvN-Local", "metric": 90},
        {"repeat_index": 0, "algorithm_core": "rscf", "scope": "PHASE_LOCAL", "treatment_name": "RSCF-Local", "metric": 95},
        {"repeat_index": 0, "algorithm_core": "oracle", "scope": "PHASE_LOCAL", "treatment_name": "Oracle-Local", "metric": 92},
        {"repeat_index": 0, "algorithm_core": "rscf", "scope": "WINDOW_JOINT", "information": "PERFECT_P2", "treatment_name": "RSCF-Joint-Perfect", "metric": 80},
        {"repeat_index": 0, "algorithm_core": "oracle", "scope": "WINDOW_JOINT", "treatment_name": "Oracle-Joint", "metric": 81},
    )
    records = _oracle_validation_records(rows, primary_metric="metric")
    by_kind = {record["oracle_validation_kind"]: record for record in records}
    assert by_kind["LOCAL"]["status"] == "INVALID_ORACLE_REFERENCE"
    assert by_kind["JOINT"]["status"] == "INVALID_ORACLE_REFERENCE"
    assert not by_kind["LOCAL"]["oracle_paper_eligible"]
    assert not by_kind["JOINT"]["oracle_paper_eligible"]


def test_trace_kind_filter_separates_measured_and_projected(tmp_path: Path) -> None:
    from rs_sim.app.formal_sweep import (
        DiscoveredFixture,
        _filter_discovered_fixtures,
    )

    measured = DiscoveredFixture(
        source_root=tmp_path,
        discovery_root=tmp_path,
        fixture_path=tmp_path / "measured" / "EP8" / "fixtures" / "m.json",
        relative_path="measured/EP8/fixtures/m.json",
    )
    projected = DiscoveredFixture(
        source_root=tmp_path,
        discovery_root=tmp_path,
        fixture_path=tmp_path / "projected" / "EP16" / "fixtures" / "p.json",
        relative_path="projected/EP16/fixtures/p.json",
    )
    fixtures = (measured, projected)
    assert _filter_discovered_fixtures(fixtures, trace_kind="measured") == (measured,)
    assert _filter_discovered_fixtures(fixtures, trace_kind="projected") == (projected,)
    assert _filter_discovered_fixtures(fixtures, trace_kind="all") == fixtures


def test_formal_presets_separate_main_and_oracle_references() -> None:
    from rs_sim.app.formal_runner import MAIN, ORACLE, PAPER

    main_names = {row["name"] for row in MAIN}
    assert {
        "FIFO-Local",
        "Greedy-Local",
        "Birkhoff-Local",
        "iSLIP-Local",
        "Residual-MWM-Local",
        "FAST-Local",
        "Aurora-Local",
        "RSCF-Local",
        "RSCF-Joint-Zero",
        "RSCF-Joint-FATE",
        "RSCF-Joint-Perfect",
    } == main_names
    oracle_names = {row["name"] for row in ORACLE}
    assert {
        "FIFO-Local",
        "Birkhoff-Local",
        "RSCF-Local",
        "RSCF-Joint-Perfect",
        "Oracle-Local",
        "Oracle-Joint",
    } == oracle_names
    assert {"Oracle-Local", "Oracle-Joint"}.isdisjoint(main_names)
    assert {"Oracle-Local", "Oracle-Joint"}.issubset(
        {row["name"] for row in PAPER}
    )
