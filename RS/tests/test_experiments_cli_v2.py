from __future__ import annotations

import json
import os
from pathlib import Path

from rs.experiments.cli import main
from rs.experiments.config_loader import ExperimentConfigLoader, UnsupportedLegacyExperimentConfig
from rs.experiments.output_schema import initialize_run_artifacts
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


def test_diagnostic_runner_returns_typed_success_bundle() -> None:
    registry = RunnerRegistry()
    plan = registry.resolve(RunKind.DIAGNOSTIC).run(
        type(
            "Plan",
            (),
            {
                "suite_id": "cpu-core",
                "case_id": "diag-case",
                "run_kind": RunKind.DIAGNOSTIC,
                "config_digest": "cfg",
                "planning_case": type(
                    "Case",
                    (),
                    {
                        "prediction_mode": "none",
                        "planner_id": "diag",
                        "planner_family": "diagnostic",
                        "execution_backend": "diagnostic",
                        "instrumentation_mode": "off",
                    },
                )(),
            },
        )()
    )
    assert plan.status == "success"
    assert plan.run_identity.pipeline == "online"
    assert plan.summary["all_work_completed"] is True
    assert plan.eligibility.correctness_eligible is True
    assert plan.eligibility.performance_eligible is False
    assert "diagnostic_mode" in plan.eligibility.reasons


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

    output_dir = tmp_path / "runs"
    assert main(["run", "--config", str(config_path), "--suite-id", "cpu-core", "--output-dir", str(output_dir)]) == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["status"] == "success"
    assert len(run_payload["runs"]) == 2
    assert all(Path(item["result_bundle_path"]).is_file() for item in run_payload["runs"])


def test_loader_rejects_legacy_v1_config_without_lossy_migration(tmp_path: Path) -> None:
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
    try:
        ExperimentConfigLoader().load(config_path=path)
    except UnsupportedLegacyExperimentConfig as exc:
        assert "schema v1 experiment config is unsupported" in str(exc)
    else:
        raise AssertionError("expected legacy v1 config rejection")


def test_official_v2_configs_preserve_model_topology_workload_and_strategy_semantics() -> None:
    loader = ExperimentConfigLoader()
    base = Path(__file__).resolve().parents[1] / "configs" / "official"
    expected = {
        "offline_evaluation.yaml": "OFFLINE_EVALUATION",
        "gloo_functional.yaml": "GLOO_FUNCTIONAL",
        "gpu_correctness.yaml": "GPU_CORRECTNESS",
        "gpu_performance.yaml": "GPU_PERFORMANCE",
        "multinode_correctness.yaml": "MULTINODE_CORRECTNESS",
        "multinode_performance.yaml": "MULTINODE_PERFORMANCE",
    }
    for name, run_kind in expected.items():
        loaded = loader.load(config_path=base / name)
        assert loaded.spec.schema_version == 2
        case = loaded.spec.planning_cases[0]
        assert case.run_kind.value == run_kind
        assert loaded.spec.defaults["model"]["id"]
        assert loaded.spec.defaults["topology"]["world_size"] > 0
        assert loaded.spec.defaults["workload"]["id"]
        assert loaded.spec.defaults["evaluation"]["repeats"] >= 1
        assert loaded.spec.defaults["evaluation"]["warmup"] >= 0
        assert len(loaded.spec.defaults["runtime"]["selected_layers"]) >= 1


def test_initialize_run_artifacts_uses_env_commit_sha_without_git(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "archive"
    repo_root.mkdir()
    output_dir = tmp_path / "run"
    monkeypatch.setenv("ROUTERSENSE_COMMIT_SHA", "env-sha-123")
    layout = initialize_run_artifacts(
        repo_root=repo_root,
        output_dir=output_dir,
        run_type="offline",
        official_entrypoint="offline_evaluation",
        config_snapshot={
            "schema_version": 1,
            "runtime": {"invariant_mode": "diagnostic", "line": "offline_replay"},
            "traffic": {"bucket_mode": "dynamic_current", "bucket_rows": 0},
            "policy": {"name": ""},
            "prediction": {"name": ""},
            "topology": {"world_size": 1},
            "model": {"id": "archive"},
            "workload": {"id": "smoke"},
        },
    )
    manifest = json.loads((layout.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit_sha"] == "env-sha-123"
    assert manifest["commit_sha_source"] == "env"
    assert manifest["source_archive_digest"] == ""
    monkeypatch.delenv("ROUTERSENSE_COMMIT_SHA", raising=False)


def test_initialize_run_artifacts_uses_handoff_manifest_sha_without_git(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ROUTERSENSE_COMMIT_SHA", raising=False)
    repo_root = tmp_path / "archive" / "RS"
    repo_root.mkdir(parents=True)
    handoff_dir = repo_root.parent / "handoff"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "manifest.json").write_text(
        json.dumps({"final_sha": "manifest-sha-456"}, ensure_ascii=True),
        encoding="utf-8",
    )
    layout = initialize_run_artifacts(
        repo_root=repo_root,
        output_dir=tmp_path / "run",
        run_type="gloo",
        official_entrypoint="gloo_functional",
        config_snapshot={
            "schema_version": 1,
            "runtime": {"invariant_mode": "diagnostic", "line": "phase_sync"},
            "traffic": {"bucket_mode": "dynamic_current", "bucket_rows": 0},
            "policy": {"name": ""},
            "prediction": {"name": ""},
            "topology": {"world_size": 1},
            "model": {"id": "archive"},
            "workload": {"id": "smoke"},
        },
    )
    manifest = json.loads((layout.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit_sha"] == "manifest-sha-456"
    assert manifest["commit_sha_source"] == "handoff_manifest"
    assert manifest["source_archive_digest"]


def test_gloo_runner_returns_non_placeholder_bundle() -> None:
    registry = RunnerRegistry()
    plan = type(
        "Plan",
        (),
        {
            "experiment_id": "official-gloo-functional",
            "suite_id": "gloo-functional",
            "case_id": "gloo-functional-core",
            "run_kind": RunKind.GLOO_FUNCTIONAL,
            "config_digest": "cfg",
            "commit_sha": "",
            "defaults": {"topology": {"world_size": 4}},
            "planning_case": type(
                "Case",
                (),
                {
                    "prediction_mode": "none",
                    "planner_id": "routersense_joint_phase_sync",
                    "planner_family": "joint",
                    "execution_backend": "gloo_functional",
                    "instrumentation_mode": "contract",
                    "predictor_id": "none",
                },
            )(),
        },
    )()
    result = registry.resolve(RunKind.GLOO_FUNCTIONAL).run(plan)
    assert result.status == "success"
    assert result.details["gate_status"] == "passed"
    assert Path(str(result.details["gate_summary_artifact_path"])).is_file()
    assert result.summary["all_work_completed"] is True
