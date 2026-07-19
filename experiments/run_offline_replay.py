#!/usr/bin/env python3
"""Unified offline replay entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

import yaml

from rs.core.formal_config_loader import load_formal_config
from rs.core.contracts.result import OFFLINE_PIPELINE, RunIdentity
from rs.experiments.output_schema import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    initialize_run_artifacts,
    read_manifest,
    update_status,
    write_json,
    write_resolved_configs,
    write_layout_result_bundle,
)
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle
from rs.runtime.guards.artifact import write_failure_artifact
from rs.runtime.guards.errors import RouterSenseInvariantError
from rs.runtime.guards import InvariantContext, require_invariant
from rs.runtime.offline.replay_unified import CanonicalBucketizer, PlanningHint, ReplayEngine, ReplayWindow, _matrix
from rs.scheduling.catalog import resolve_algorithm_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args()
def _load_windows(fixture_dir: Path, *, max_windows: int | None = None) -> list[ReplayWindow]:
    windows: list[ReplayWindow] = []
    for path in sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda item: int(item.stem.split("_")[-1])):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata", {}) or {})
        windows.append(
            ReplayWindow(
                fixture_id=str(path.stem),
                window_id=f"{metadata.get('layer_id', path.stem)}->{metadata.get('next_layer_id', '')}",
                layer_id=int(metadata.get("layer_id", 0)),
                p0_truth_rows=_matrix(payload["p0_dispatch_matrix"]),
                p1_truth_rows=_matrix(payload["p1_return_matrix"]),
                p2_truth_rows=_matrix(payload.get("p2_next_dispatch_matrix", payload.get("p2_next_dispatch_forecast_matrix", []))),
                matrix_unit="rows",
                group_size=len(payload["p0_dispatch_matrix"]),
                payload_row_bytes_by_phase={"P0": 1, "P1": 1, "P2": 1},
                metadata=metadata,
            )
        )
        if max_windows is not None and len(windows) >= int(max_windows):
            break
    return windows


def _digest_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hint(window: ReplayWindow, hint_type: str) -> PlanningHint:
    if hint_type == "zero_hint":
        matrix = tuple(tuple(0 for _ in row) for row in window.p2_truth_rows)
        confidence = 0.0
        source_layer = None
    elif hint_type == "copy_current_dispatch":
        matrix = window.p0_truth_rows
        confidence = 1.0
        source_layer = int(window.layer_id)
    elif hint_type == "perfect_trace_hint":
        matrix = window.p2_truth_rows
        confidence = 1.0
        source_layer = int(window.layer_id)
    else:
        raise ValueError(f"unsupported hint_type {hint_type!r}")
    return PlanningHint(
        hint_type=hint_type,
        p2_hint_rows=_matrix(matrix),
        confidence=float(confidence),
        source_layer=source_layer,
        target_layer=int(window.layer_id) + 1,
    )


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    layout = None
    try:
        resolved = load_formal_config(
            config_path=config_path,
            expected_runtime_line="offline_replay",
            official_entrypoint="experiments/run_offline_replay.py",
        )
        config = resolved.normalized_config
        replay_cfg = dict(config.get("replay", {}) or {})
        default_output_dir = ROOT / str(replay_cfg.get("output_dir", "outputs/offline/offline_replay_smoke"))
        output_dir = (ROOT / str(args.output_dir)).resolve() if args.output_dir else default_output_dir.resolve()
        layout = initialize_run_artifacts(
            repo_root=ROOT,
            output_dir=output_dir,
            run_type="offline",
            official_entrypoint="experiments/run_offline_replay.py",
            config_snapshot=config,
        )
        write_resolved_configs(
            layout,
            normalized_config=config,
            consumed_config=config,
        )
        fixture_dir = (ROOT / str(replay_cfg["fixture_dir"])).resolve()
        evaluation = dict(config.get("evaluation", {}) or {})
        traffic = dict(config.get("traffic", {}) or {})
        policy_cfg = dict(config.get("policy", {}) or {})
        prediction_cfg = dict(config.get("prediction", {}) or {})
        windows = _load_windows(fixture_dir, max_windows=int(evaluation.get("max_windows", 4)))
        bucket_rows_values = [int(value) for value in traffic.get("bucket_rows", [1024])]
        policies = [str(item) for item in policy_cfg.get("names", [])]
        hints = [str(item) for item in prediction_cfg.get("names", [])]
        rows: list[dict[str, Any]] = []
        invariant_rows: list[dict[str, Any]] = []
        for bucket_rows in bucket_rows_values:
            bucketizer = CanonicalBucketizer(bucket_rows=bucket_rows)
            engine = ReplayEngine(
                scheduling_mode=str(replay_cfg.get("scheduling_mode", "execution_window")),
                expert_compute_delay=float(replay_cfg.get("expert_compute_delay", 0.0)),
                bucket_rows=bucket_rows,
                max_waves=int(replay_cfg.get("max_waves", evaluation.get("max_waves", 256))),
            )
            for window in windows:
                tasks = bucketizer.bucketize(window)
                task_digest = CanonicalBucketizer.digest(tasks)
                for policy_name in policies:
                    resolved_policy = resolve_algorithm_id(policy_name)
                    for hint_type in hints:
                        result = engine.execute(
                            replay_window=window,
                            planning_hint=_hint(window, hint_type),
                            policy_name=resolved_policy.builder_key,
                        )
                        rows.append(
                            {
                                "fixture_id": window.fixture_id,
                                "window_id": window.window_id,
                                "layer_id": window.layer_id,
                                "bucket_rows": bucket_rows,
                                "requested_policy_name": policy_name,
                                "canonical_policy_name": resolved_policy.canonical_name,
                                "builder_policy_name": resolved_policy.builder_key,
                                "hint_type": hint_type,
                                "input_task_count": result["input_task_count"],
                                "input_total_rows": result["input_total_rows"],
                                "input_task_digest": result["input_task_digest"],
                                "logical_plan_digest": result["logical_plan_digest"],
                                "makespan": result["makespan"],
                                "audit_valid": result["audit_valid"],
                            }
                        )
                invariant_rows.append(
                    {
                        "fixture_id": window.fixture_id,
                        "window_id": window.window_id,
                        "bucket_rows": bucket_rows,
                        "input_task_count": len(tasks),
                        "input_total_rows": int(sum(task.row_count for task in tasks)),
                        "input_task_digest": task_digest,
                    }
                )
        audit_invalid_count = int(sum(1 for row in rows if not bool(row.get("audit_valid", False))))
        require_invariant(
            audit_invalid_count == 0,
            context=InvariantContext(stage="offline", error_code="RS-OFFLINE-001"),
            message="offline replay produced invalid audit rows",
            expected=0,
            actual=audit_invalid_count,
        )
        manifest = read_manifest(layout)
        summary_payload = {
            "rows": rows,
            "invariants": invariant_rows,
            "run_valid": True,
            "audit_invalid_count": audit_invalid_count,
            "failure_codes": [],
            "commit_sha": manifest["commit_sha"],
        }
        write_json(output_dir / "summary.json", summary_payload)
        write_json(layout.metrics_dir / "summary.json", summary_payload)
        run_kind = "OFFLINE_EVALUATION"
        future_information_mode = "predicted"
        summary = {
            "run_kind": run_kind,
            "all_work_completed": True,
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 0,
            "cleanup_failure_count": 0,
            "execution_outcome_count": 0,
            "missing_execution_outcome_count": 0,
            "formal_execution_expected": False,
            "offline_replay_complete": True,
            "evaluation_spec_digest": _digest_payload(
                {
                    "fixture_dir": str(fixture_dir),
                    "bucket_rows": bucket_rows_values,
                    "policies": policies,
                    "hints": hints,
                    "max_windows": int(evaluation.get("max_windows", 4)),
                    "scheduling_mode": str(replay_cfg.get("scheduling_mode", "execution_window")),
                    "expert_compute_delay": float(replay_cfg.get("expert_compute_delay", 0.0)),
                }
            ),
            "task_set_digest": _digest_payload(invariant_rows),
            "execution_truth_digest": _digest_payload(
                [
                    {
                        "fixture_id": window.fixture_id,
                        "window_id": window.window_id,
                        "layer_id": window.layer_id,
                        "p0_truth_rows": window.p0_truth_rows,
                        "p1_truth_rows": window.p1_truth_rows,
                        "p2_truth_rows": window.p2_truth_rows,
                    }
                    for window in windows
                ]
            ),
            "offline_record_count": len(rows),
            "offline_audit_status": "valid",
            "coverage_status": "complete",
            "performance_measurement_complete": False,
            "measured_repeat_count": 0,
            "warmup_excluded": False,
            "preparation_miss_count": 0,
            "provisional_execution_count": 0,
            "materialization_failure_count": 0,
            "execution_failure_count": 0,
            "native_fallback_count": 0,
            "semantic_failure_fallback_count": 0,
            "safe_selector_fallback_count": 0,
        }
        details = {
            "run_kind": run_kind,
            "offline_fixture_dir": str(fixture_dir),
            "bucket_rows": list(bucket_rows_values),
            "policy_names": list(policies),
            "hint_names": list(hints),
            "window_count": len(windows),
            "row_count": len(rows),
            "invariant_count": len(invariant_rows),
        }
        result_bundle = build_result_bundle(
            ResultBundleDraft(
                run_identity=RunIdentity(
                    run_id=str(layout.root.name),
                    pipeline=OFFLINE_PIPELINE,
                    claim_scope="offline_replay",
                    trace_origin="fixture",
                    future_information_mode=future_information_mode,
                ),
                status="success",
                correctness_status="valid",
                performance_status="ineligible",
                commit_sha=str(manifest["commit_sha"]),
                git_clean=not bool(manifest.get("git_dirty", False)),
                instrumentation_mode="off",
                audit_evidence_level="summary_only",
                measurement_complete=False,
                summary=summary,
                details=details,
                extensions={},
            )
        )
        write_layout_result_bundle(layout, result_bundle)
        update_status(
            layout,
            status=RUN_STATUS_COMPLETED,
            extra={
                "row_count": len(rows),
                "invariant_count": len(invariant_rows),
                "audit_invalid_count": audit_invalid_count,
                "result_bundle_path": "result_bundle.json",
                "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
    except RouterSenseInvariantError as exc:
        if layout is not None:
            write_failure_artifact(layout.failures_dir / "offline_invariant_failure.json", error=exc)
            update_status(
                layout,
                status=RUN_STATUS_FAILED,
                extra={
                    "failure_codes": [exc.failure.error_code],
                    "run_valid": False,
                },
            )
        print(f"[offline_replay] invariant failure: {exc.failure.error_code}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        if layout is not None:
            failure_payload = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
            (layout.failures_dir / "offline_exception.json").write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            update_status(layout, status=RUN_STATUS_FAILED, extra={"run_valid": False})
        print(f"[offline_replay] exception: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
