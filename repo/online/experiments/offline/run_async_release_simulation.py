#!/usr/bin/env python3
"""Run async-release CPU simulation over replay fixtures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.offline.prediction import rolling_predictor_records
from rs.runtime.online.megatron_ep.async_release import simulate_async_release
from experiments.offline.replay_fixture_policy_study import _build_problem
from rs.scheduling import resolve_policy
from rs.scheduling.online_adapters import build_priority_artifact_from_plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fixture_dir = Path(args.fixture_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_linear")
    predicted_by_layer = {str(row.layer_id): row.predicted_matrix for row in prediction_rows}
    rows = []
    for fixture_path in sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        layer_id = str(fixture.get("metadata", {}).get("layer_id", ""))
        predicted = predicted_by_layer.get(layer_id)
        if predicted is None:
            predicted = tuple(tuple(int(value) for value in row) for row in fixture["p0_dispatch_matrix"])
        problem = _build_problem(
            fixture,
            mode="runtime_lookahead",
            p2_source="fate_style_linear" if layer_id in predicted_by_layer else "copy_current_dispatch",
            expert_compute_delay=float(args.expert_compute_delay),
            predicted_p2_matrix=predicted,
        )
        safe_plan = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0).build_logical_plan(problem)
        artifact = build_priority_artifact_from_plan(
            problem=problem,
            plan=safe_plan,
            heuristic_family="barrier_criticality_matching",
            predictor_name="fate_style_linear" if layer_id in predicted_by_layer else "copy_current_dispatch",
            p2_source="fate_style_linear" if layer_id in predicted_by_layer else "copy_current_dispatch",
        )
        sim = simulate_async_release(
            p0_dispatch_matrix=fixture["p0_dispatch_matrix"],
            p1_return_matrix=fixture["p1_return_matrix"],
            predicted_p2_matrix=predicted,
            compute_delay=float(args.expert_compute_delay),
            policy_name="routersense_joint_async_release",
            priority_artifact=artifact,
        )
        rows.append({"fixture_name": fixture_path.name, "layer_id": layer_id, **sim})
    summary = {
        "fixture_dir": str(fixture_dir),
        "rows": rows,
        "mean_completion_time": statistics.mean([float(row["completion_time"]) for row in rows]) if rows else 0.0,
        "mean_hidden_planning_fraction": statistics.mean([float(row["hidden_planning_fraction"]) for row in rows]) if rows else 0.0,
        "total_dependency_violations": sum(int(row["dependency_violations"]) for row in rows),
        "total_fallback_replan_count": sum(int(row["fallback_replan_count"]) for row in rows),
    }
    (output_dir / "async_release_sim_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "async_release_sim_summary.md").write_text(
        "\n".join(
            [
                "# Async Release Simulation",
                "",
                f"- fixture_dir: `{fixture_dir}`",
                f"- mean_completion_time: {summary['mean_completion_time']}",
                f"- mean_hidden_planning_fraction: {summary['mean_hidden_planning_fraction']}",
                f"- total_dependency_violations: {summary['total_dependency_violations']}",
                f"- total_fallback_replan_count: {summary['total_fallback_replan_count']}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
