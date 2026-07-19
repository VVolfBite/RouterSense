from __future__ import annotations

import json
from pathlib import Path

from rs.core.contracts.result import RunIdentity
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle
from rs.reporting.prepared_plan_runtime_analysis import analyze_prepared_plan_runtime


def _write_result_bundle(run_dir: Path, *, details: dict[str, object]) -> None:
    bundle = build_result_bundle(
        ResultBundleDraft(
            run_identity=RunIdentity(
                run_id="run",
                pipeline="online",
                claim_scope="formal",
                trace_origin="runtime",
                future_information_mode="predicted",
            ),
            status="success",
            correctness_status="valid",
            performance_status="ineligible",
            commit_sha="abc123",
            git_clean=True,
            instrumentation_mode="contract",
            audit_evidence_level="summary_only",
            measurement_complete=True,
            summary={
                "all_work_completed": True,
                "fallback_count": 0,
                "timeout_count": 0,
                "check_failure_count": 0,
                "execution_outcome_count": 1,
            },
            details=details,
            extensions={},
        )
    )
    (run_dir / "result_bundle.json").write_text(json.dumps(bundle.to_dict()), encoding="utf-8")


def test_analyze_prepared_plan_runtime_extracts_arrival_binding_and_plan_summary(tmp_path: Path) -> None:
    run_dir = tmp_path
    _write_result_bundle(
        run_dir,
        details={
            "policy_name": "routersense_p0p1p2_hint",
            "p2_hint_mode": "calibrated_artifact",
            "execution_audit_status": "passed",
        },
    )
    (run_dir / "rank0_plan_arrival_records.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"arrival_status": "none", "has_prepared_plan": False, "plan_age_us": 0}),
                json.dumps(
                    {
                        "arrival_status": "before_commit",
                        "has_prepared_plan": True,
                        "plan_age_us": 123,
                        "source_layer": "model.layers.0.mlp",
                        "window_key": "window-1",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "rank0_prepared_plan_bindings.jsonl").write_text(
        json.dumps({"source_layer_name": "model.layers.0.mlp", "target_layer_id": "1", "window_key": "window-1"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "rank0_scheduled_phase_plans.jsonl").write_text(
        json.dumps(
            {
                "plan_key": {"layer_id": "1"},
                "phase": "P0",
                "plan_hash": "plan-a",
                "waves": [{"wave_id": 0, "bucket_tasks": [{"task_id": "P0:0->1:bucket:0", "src_rank": 0, "dst_rank": 1, "row_count": 16, "byte_count": 65536}]}],
                "metrics": {
                    "prepared_window_key": "window-1",
                    "source_logical_plan_hash": "source-hash",
                    "ordered_by_prepared_plan": True,
                    "hint_edges_available": 2,
                    "hint_edges_matched": 2,
                    "hint_edges_consumed": 2,
                    "hint_match_rate": 1.0,
                    "bucket_count": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "rank0_execution_audit.json").write_text(
        json.dumps(
            {
                "audits": [
                    {
                        "layer_id": "1",
                        "phase": "P0",
                        "status": "passed",
                        "planned_wave_count": 1,
                        "executed_wave_count": 1,
                        "p0_bundle_atomicity_preserved": True,
                        "details": {
                            "hint_edges_consumed": 2,
                            "hint_match_rate": 1.0,
                            "prepared_window_key": "window-1",
                            "source_logical_plan_hash": "source-hash",
                            "prepared_plan_order_preserved": True,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_prepared_plan_runtime(run_dir, rank=0)
    assert report["policy_name"] == "routersense_p0p1p2_hint"
    assert report["plan_arrival_summary"]["prepared_plan_arrival_count"] == 1
    assert report["plan_arrival_summary"]["before_commit_count"] == 1
    assert report["prepared_plan_bindings"][0]["window_key"] == "window-1"
    assert report["scheduled_phase_plan_summaries"][0]["ordered_by_prepared_plan"] is True
    assert report["scheduled_phase_plan_summaries"][0]["hint_edges_consumed"] == 2
    assert report["execution_audit_summaries"][0]["prepared_plan_order_preserved"] is True


def test_analyze_prepared_plan_runtime_requires_canonical_result_bundle(tmp_path: Path) -> None:
    try:
        analyze_prepared_plan_runtime(tmp_path, rank=0)
    except FileNotFoundError as exc:
        assert "missing canonical result bundle" in str(exc)
    else:
        raise AssertionError("expected missing canonical result bundle failure")
