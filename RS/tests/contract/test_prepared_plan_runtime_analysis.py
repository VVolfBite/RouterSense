from __future__ import annotations

import json
from pathlib import Path

from rs.reporting.prepared_plan_runtime_analysis import analyze_prepared_plan_runtime


def test_analyze_prepared_plan_runtime_extracts_arrival_binding_and_plan_summary(tmp_path: Path) -> None:
    run_dir = tmp_path
    (run_dir / "summary.json").write_text(
        json.dumps({"details": {"policy_name": "routersense_p0p1p2_hint", "p2_hint_mode": "calibrated_artifact", "execution_audit_status": "passed"}}),
        encoding="utf-8",
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
