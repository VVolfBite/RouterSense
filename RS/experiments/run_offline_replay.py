#!/usr/bin/env python3
"""Unified offline replay entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

import yaml

from rs.runtime.offline.replay_unified import CanonicalBucketizer, PlanningHint, ReplayEngine, ReplayWindow, _matrix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
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
    config = _load_yaml(config_path)
    output_dir = (ROOT / str(config.get("output_dir", "outputs/offline/offline_replay_smoke"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = (ROOT / str(config["fixture_dir"])).resolve()
    windows = _load_windows(fixture_dir, max_windows=int(config.get("max_windows", 4)))
    bucket_rows_values = [int(value) for value in config.get("bucket_rows", [1024])]
    policies = [str(item) for item in config.get("policies", [])]
    hints = [str(item) for item in config.get("hints", [])]
    rows: list[dict[str, Any]] = []
    invariant_rows: list[dict[str, Any]] = []
    for bucket_rows in bucket_rows_values:
        bucketizer = CanonicalBucketizer(bucket_rows=bucket_rows)
        engine = ReplayEngine(
            scheduling_mode=str(config.get("scheduling_mode", "execution_window")),
            expert_compute_delay=float(config.get("expert_compute_delay", 0.0)),
            bucket_rows=bucket_rows,
        )
        for window in windows:
            tasks = bucketizer.bucketize(window)
            task_digest = CanonicalBucketizer.digest(tasks)
            for policy_name in policies:
                for hint_type in hints:
                    result = engine.execute(
                        replay_window=window,
                        planning_hint=_hint(window, hint_type),
                        policy_name=policy_name,
                    )
                    rows.append(
                        {
                            "fixture_id": window.fixture_id,
                            "window_id": window.window_id,
                            "layer_id": window.layer_id,
                            "bucket_rows": bucket_rows,
                            "policy_name": policy_name,
                            "hint_type": hint_type,
                            "input_task_count": result["input_task_count"],
                            "input_total_rows": result["input_total_rows"],
                            "input_task_digest": result["input_task_digest"],
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
    (output_dir / "summary.json").write_text(json.dumps({"rows": rows, "invariants": invariant_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
