from __future__ import annotations

import json
from pathlib import Path

from rs.experiments.cli import main
from rs.experiments.config_loader import ExperimentConfigLoader
from rs.experiments.registry import RunnerRegistry
from rs.experiments.specs import RunKind


def _write_config(path: Path) -> Path:
    path.write_text(
        """
schema_version: 2
experiment_id: cpu-core
suites:
  - suite_id: cpu-core
    markers: [unit, integration_cpu]
    case_ids: [diag-case, offline-case]
    run_kinds: [DIAGNOSTIC, OFFLINE_EVALUATION]
planning_cases:
  - case_id: diag-case
    run_kind: DIAGNOSTIC
    planner_id: diag
    planner_family: diagnostic
    selector_mode: fixed
    predictor_id: none
    prediction_mode: none
    execution_backend: diagnostic
    instrumentation_mode: off
    fallback_policy: fail_closed
  - case_id: offline-case
    run_kind: OFFLINE_EVALUATION
    planner_id: fifo
    planner_family: baseline
    selector_mode: fixed
    predictor_id: zero
    prediction_mode: zero
    execution_backend: simulator
    instrumentation_mode: perf_light
    fallback_policy: fail_closed
defaults:
  seed: 7
""".strip(),
        encoding="utf-8",
    )
    return path


def test_loader_validates_schema_v2_and_writes_resolved_artifacts(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "experiment.yaml")
    loaded = ExperimentConfigLoader().load(config_path=config_path)
    assert loaded.spec.experiment_id == "cpu-core"
    assert loaded.spec.schema_version == 2
    out = tmp_path / "resolved"
    ExperimentConfigLoader().write_resolved_artifacts(loaded, output_dir=out)
    assert (out / "resolved_config.yaml").exists()
    assert (out / "migration_report.json").exists()


def test_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema_version: 2
experiment_id: bad
bad_field: true
suites: []
planning_cases: []
""".strip(),
        encoding="utf-8",
    )
    try:
        ExperimentConfigLoader().load(config_path=path)
    except ValueError as exc:
        assert "unknown top-level" in str(exc)
    else:
        raise AssertionError("expected unknown field validation failure")


def test_registry_lists_expected_run_kinds() -> None:
    kinds = set(RunnerRegistry().list_run_kinds())
    assert RunKind.DIAGNOSTIC.value in kinds
    assert RunKind.OFFLINE_EVALUATION.value in kinds
    assert RunKind.GLOO_FUNCTIONAL.value in kinds


def test_cli_inspect_plan_run_and_list(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path / "experiment.yaml")
    assert main(["inspect-config", "--config", str(config_path)]) == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["schema_version"] == 2

    assert main(["plan", "--config", str(config_path)]) == 0
    plan_payload = json.loads(capsys.readouterr().out)
    assert len(plan_payload) == 2

    assert main(["list-suites", "--config", str(config_path)]) == 0
    suites_payload = json.loads(capsys.readouterr().out)
    assert suites_payload[0]["suite_id"] == "cpu-core"

    assert main(["list-cases", "--config", str(config_path)]) == 0
    cases_payload = json.loads(capsys.readouterr().out)
    assert {case["case_id"] for case in cases_payload} == {"diag-case", "offline-case"}

    assert main(["run", "--config", str(config_path), "--suite-id", "cpu-core"]) == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert len(run_payload) == 2
    assert {item["status"] for item in run_payload} == {"invalid"}


def test_loader_migrates_v1_single_case_config(tmp_path: Path) -> None:
    path = tmp_path / "v1.yaml"
    path.write_text(
        """
schema_version: 1
name: migrated
case_id: legacy-case
run_kind: DIAGNOSTIC
planner_id: none
planner_family: none
selector_mode: fixed
predictor_id: none
prediction_mode: none
execution_backend: diagnostic
instrumentation_mode: off
fallback_policy: fail_closed
""".strip(),
        encoding="utf-8",
    )
    loaded = ExperimentConfigLoader().load(config_path=path)
    assert loaded.spec.schema_version == 2
    assert loaded.migration_report["migrated"] is True
    assert loaded.spec.planning_cases[0].case_id == "legacy-case"
