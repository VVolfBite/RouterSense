from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs.experiments.output_schema import validate_official_entrypoint_config
from rs.core.contracts.result import ResultBundle, RunIdentity
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle
from rs.reporting.schema import validate_report_eligibility
from rs.runtime.guards.errors import RuntimeStateFieldError
from rs.runtime.online.megatron_ep.state.runtime_state import PreparedWindowRuntimeState


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_result_bundle(path: Path, summary: dict, *, status: str = "success") -> None:
    bundle = build_result_bundle(
        ResultBundleDraft(
            run_identity=RunIdentity(
                run_id="run",
                pipeline="offline",
                claim_scope="formal",
                trace_origin="fixture",
                future_information_mode="predicted",
            ),
            status=status,
            correctness_status="valid" if status == "success" else "invalid",
            performance_status="ineligible",
            commit_sha="abc",
            git_clean=True,
            instrumentation_mode="contract",
            audit_evidence_level="summary_only",
            measurement_complete=True,
            summary=summary,
            details={},
            extensions={},
        )
    )
    path.write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def test_runtime_state_unknown_field_hard_fails_in_strict_mode() -> None:
    state = PreparedWindowRuntimeState()
    state.set_invariant_mode("evaluation_strict")

    with pytest.raises(RuntimeStateFieldError):
        state.write("typo_metric_name", 1)
    with pytest.raises(RuntimeStateFieldError):
        state.read("typo_metric_name")
    with pytest.raises(RuntimeStateFieldError):
        state.pop("typo_metric_name")


def test_validate_official_entrypoint_config_rejects_non_power_of_two_bucket() -> None:
    with pytest.raises(Exception):
        validate_official_entrypoint_config(
            config_snapshot={
                "schema_version": 1,
                "runtime": {"line": "offline_replay", "invariant_mode": "evaluation_strict"},
                "traffic": {"bucket_rows": [768]},
            },
            expected_runtime_line="offline_replay",
            official_entrypoint="experiments/run_offline_replay.py",
        )


def test_validate_official_entrypoint_config_accepts_power_of_two_bucket_list() -> None:
    validate_official_entrypoint_config(
        config_snapshot={
            "schema_version": 1,
            "runtime": {"line": "offline_replay", "invariant_mode": "diagnostic"},
            "traffic": {"bucket_rows": [512, 1024]},
        },
        expected_runtime_line="offline_replay",
        official_entrypoint="experiments/run_offline_replay.py",
    )


def test_validate_official_entrypoint_config_rejects_string_schema_version() -> None:
    with pytest.raises(Exception):
        validate_official_entrypoint_config(
            config_snapshot={
                "schema_version": "1",
                "runtime": {"line": "offline_replay", "invariant_mode": "diagnostic"},
                "traffic": {"bucket_rows": [512]},
            },
            expected_runtime_line="offline_replay",
            official_entrypoint="experiments/run_offline_replay.py",
        )


def test_validate_official_entrypoint_config_rejects_string_bucket_rows() -> None:
    with pytest.raises(Exception):
        validate_official_entrypoint_config(
            config_snapshot={
                "schema_version": 1,
                "runtime": {"line": "offline_replay", "invariant_mode": "diagnostic"},
                "traffic": {"bucket_rows": ["512"]},
            },
            expected_runtime_line="offline_replay",
            official_entrypoint="experiments/run_offline_replay.py",
        )


def test_validate_official_entrypoint_config_rejects_reference_only_online_policy() -> None:
    with pytest.raises(Exception):
        validate_official_entrypoint_config(
            config_snapshot={
                "schema_version": 1,
                "runtime": {"line": "async_release", "invariant_mode": "evaluation_strict"},
                "traffic": {"bucket_rows": 1024},
                "policy": {"name": "oracle_joint_cp_sat"},
            },
            expected_runtime_line="async_release",
            official_entrypoint="experiments/run_online_async_release.py",
        )


def test_report_eligibility_rejects_invalid_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "manifest.json",
        {
            "status": "completed",
            "git_dirty": False,
            "commit_sha": "abc",
        },
    )
    _write_result_bundle(
        run_dir / "result_bundle.json",
        {
            "fallback_count": 1,
            "timeout_count": 0,
            "check_failure_count": 0,
            "execution_outcome_count": 1,
            "all_work_completed": True,
        },
    )

    eligibility = validate_report_eligibility(run_dir, report_type="offline")
    assert not eligibility.eligible
    assert "fallback_count_nonzero" in eligibility.failures


def test_report_eligibility_rejects_a2_without_c2() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        _write_json(
            run_dir / "manifest.json",
            {
                "status": "completed",
                "git_dirty": False,
                "commit_sha": "abc",
            },
        )
        _write_result_bundle(
            run_dir / "result_bundle.json",
            {
                "fallback_count": 0,
                "timeout_count": 0,
                "check_failure_count": 0,
                "execution_outcome_count": 1,
                "all_work_completed": True,
            },
        )

        eligibility = validate_report_eligibility(run_dir, report_type="a2")
        assert not eligibility.eligible
        assert "performance_eligibility_false" in eligibility.failures
