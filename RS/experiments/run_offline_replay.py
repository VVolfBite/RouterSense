#!/usr/bin/env python3
"""Unified offline replay entrypoint."""

from __future__ import annotations

import argparse
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

from rs.core.config_normalization import canonical_offline_replay_payload, legacy_offline_replay_payload, normalize_run_config
from rs.experiments.output_schema import initialize_run_artifacts, update_status, validate_official_entrypoint_config, write_json
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


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


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
        normalized = normalize_run_config(_load_yaml(config_path), source_path=config_path)
        config = canonical_offline_replay_payload(normalized)
        validate_official_entrypoint_config(
            config_snapshot=config,
            expected_runtime_line="offline_replay",
            official_entrypoint="experiments/run_offline_replay.py",
        )
        legacy_config = legacy_offline_replay_payload(normalized)
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
        summary_payload = {
            "rows": rows,
            "invariants": invariant_rows,
            "run_valid": True,
            "valid_for_evaluation": True,
            "audit_invalid_count": audit_invalid_count,
            "failure_codes": [],
            "commit_sha": json.loads((layout.root / "manifest.json").read_text(encoding="utf-8"))["commit_sha"],
        }
        write_json(output_dir / "summary.json", summary_payload)
        write_json(layout.metrics_dir / "summary.json", summary_payload)
        (output_dir / "legacy_config_snapshot.yaml").write_text(yaml.safe_dump(legacy_config, sort_keys=False), encoding="utf-8")
        (layout.raw_dir / "legacy_config_snapshot.yaml").write_text(yaml.safe_dump(legacy_config, sort_keys=False), encoding="utf-8")
        update_status(
            layout,
            status="completed",
            extra={
                "row_count": len(rows),
                "invariant_count": len(invariant_rows),
                "audit_invalid_count": audit_invalid_count,
                "valid_for_evaluation": True,
                "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
    except RouterSenseInvariantError as exc:
        if layout is not None:
            write_failure_artifact(layout.failures_dir / "offline_invariant_failure.json", error=exc)
            update_status(
                layout,
                status="failed",
                extra={
                    "valid_for_evaluation": False,
                    "failure_codes": [exc.failure.error_code],
                    "run_valid": False,
                },
            )
        raise SystemExit(2) from exc
    except Exception as exc:
        if layout is not None:
            failure_payload = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
            (layout.failures_dir / "offline_exception.json").write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            update_status(layout, status="failed", extra={"valid_for_evaluation": False, "run_valid": False})
        raise


if __name__ == "__main__":
    main()
