#!/usr/bin/env python3
"""Replay predicted next-dispatch matrices against next-layer scheduling outcomes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline.replay_fixture_policy_study import _build_problem
from rs.runtime.offline.prediction import rolling_predictor_records, summarize_prediction_records
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling import resolve_policy


POLICIES = (
    "RS_safe_barrier_criticality",
    "RS_safe_gated_greedy",
    "RS_safe_gated_maxweight",
)
DEFAULT_SOURCES = (
    "zero_hint",
    "copy_current_dispatch",
    "fate_style_history",
    "fate_style_linear",
    "perfect_trace",
    "actual_trace",
)
PREDICTOR_TO_SOURCE = {
    "zero_hint": "zero_hint",
    "copy_current_dispatch": "copy_current_dispatch",
    "fate_style_history": "fate_style_history",
    "fate_style_linear": "fate_style_linear",
    "perfect_trace": "perfect_trace",
    "actual_trace": "actual_trace",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--p2-sources", nargs="*", default=None)
    return parser.parse_args()


def run_prediction_replay_suite(
    *,
    fixture_dir: Path,
    policies: Iterable[str] | None = None,
    p2_sources: Iterable[str] | None = None,
) -> dict[str, Any]:
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    selected_policies = tuple(policies or POLICIES)
    selected_sources = tuple(p2_sources or DEFAULT_SOURCES)
    predictor_records = {
        "fate_style_history": {record.layer_id: record for record in rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_history")},
        "fate_style_linear": {record.layer_id: record for record in rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_linear")},
    }
    rows: list[dict[str, Any]] = []
    for fixture_path in fixture_paths:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        source_layer_id = str(fixture.get("metadata", {}).get("layer_id", ""))
        target_layer_id = str(fixture.get("metadata", {}).get("next_layer_id", ""))
        for source in selected_sources:
            predicted_p2 = None
            prediction_metrics = {
                "prediction_remote_l1_error": 0.0,
                "prediction_relative_l1_error": 0.0,
                "prediction_cosine_similarity": 1.0 if source in {"perfect_trace", "actual_trace"} else 0.0,
                "forecast_remote_bytes": 0,
                "actual_remote_bytes": 0,
                "predictor_name": source,
                "oracle_prediction": source in {"perfect_trace", "actual_trace"},
            }
            if source in predictor_records:
                record = predictor_records[source].get(source_layer_id)
                if record is not None:
                    predicted_p2 = record.predicted_matrix
                    prediction_metrics = {
                        "prediction_remote_l1_error": float(record.absolute_l1_error),
                        "prediction_relative_l1_error": float(record.relative_l1_error),
                        "prediction_cosine_similarity": float(record.cosine_similarity),
                        "forecast_remote_bytes": int(sum(sum(row) for row in record.predicted_matrix)),
                        "actual_remote_bytes": int(sum(sum(row) for row in record.actual_matrix)),
                        "predictor_name": str(record.predictor_name),
                        "oracle_prediction": False,
                    }
            problem = _build_problem(
                fixture,
                mode="runtime_lookahead",
                p2_source=PREDICTOR_TO_SOURCE[source],
                expert_compute_delay=0.0,
                predicted_p2_matrix=predicted_p2,
            )
            policy_results: list[dict[str, Any]] = []
            for policy_name in selected_policies:
                plan = resolve_policy(policy_name=policy_name, bucket_rows=0).build_logical_plan(problem)
                audit = replay_and_audit_logical_plan(problem, plan)
                policy_results.append(
                    {
                        "source_layer_id": source_layer_id,
                        "target_layer_id": target_layer_id,
                        "policy_name": policy_name,
                        "predictor_name": prediction_metrics["predictor_name"],
                        "p2_source": source,
                        "prediction_remote_l1_error": prediction_metrics["prediction_remote_l1_error"],
                        "prediction_relative_l1_error": prediction_metrics["prediction_relative_l1_error"],
                        "prediction_cosine_similarity": prediction_metrics["prediction_cosine_similarity"],
                        "forecast_remote_bytes": prediction_metrics["forecast_remote_bytes"],
                        "actual_remote_bytes": prediction_metrics["actual_remote_bytes"],
                        "safe_makespan": float(audit.get("makespan", plan.diagnostics.get("makespan", 0.0))),
                        "fallback_to_B": bool(plan.diagnostics.get("fallback_to_paired_b", False)),
                        "selected_policy": str(plan.diagnostics.get("selected_policy", plan.policy_name)),
                        "evaluation_eligible": bool(plan.diagnostics.get("evaluation_eligible", True)),
                        "oracle_prediction": bool(prediction_metrics["oracle_prediction"]),
                    }
                )
            rows.extend(policy_results)
    summary_rows: list[dict[str, Any]] = []
    for policy_name in selected_policies:
        policy_rows = [row for row in rows if row["policy_name"] == policy_name]
        grouped = {source: [row for row in policy_rows if row["p2_source"] == source] for source in selected_sources}
        zero_by_layer = {row["source_layer_id"]: float(row["safe_makespan"]) for row in grouped["zero_hint"]}
        perfect_key = "perfect_trace" if "perfect_trace" in grouped else ("actual_trace" if "actual_trace" in grouped else None)
        perfect_by_layer = {row["source_layer_id"]: float(row["safe_makespan"]) for row in grouped.get(perfect_key or "", [])}
        for source, source_rows in grouped.items():
            makespans = [float(row["safe_makespan"]) for row in source_rows]
            relative_to_zero = []
            gap_to_perfect = []
            for row in source_rows:
                lid = row["source_layer_id"]
                if lid in zero_by_layer and zero_by_layer[lid] > 0:
                    relative_to_zero.append((float(row["safe_makespan"]) - zero_by_layer[lid]) / zero_by_layer[lid])
                if lid in perfect_by_layer and perfect_by_layer[lid] > 0:
                    gap_to_perfect.append((float(row["safe_makespan"]) - perfect_by_layer[lid]) / perfect_by_layer[lid])
            summary_rows.append(
                {
                    "policy_name": policy_name,
                    "predictor_name": source_rows[0]["predictor_name"] if source_rows else source,
                    "p2_source": source,
                    "mean_makespan": statistics.mean(makespans) if makespans else None,
                    "relative_to_zero_hint": statistics.mean(relative_to_zero) if relative_to_zero else None,
                    "gap_to_perfect_trace": statistics.mean(gap_to_perfect) if gap_to_perfect else None,
                    "fallback_to_B_ratio": (
                        sum(1 for row in source_rows if row["fallback_to_B"]) / len(source_rows) if source_rows else 0.0
                    ),
                    "selected_U_ratio": (
                        sum(1 for row in source_rows if not row["fallback_to_B"]) / len(source_rows) if source_rows else 0.0
                    ),
                    "mean_prediction_relative_l1_error": statistics.mean([float(row["prediction_relative_l1_error"]) for row in source_rows]) if source_rows else 0.0,
                    "mean_prediction_cosine_similarity": statistics.mean([float(row["prediction_cosine_similarity"]) for row in source_rows]) if source_rows else 0.0,
                    "evaluation_eligible": bool(source_rows[0]["evaluation_eligible"]) if source_rows else False,
                    "oracle_prediction": bool(source_rows[0]["oracle_prediction"]) if source_rows else False,
                }
            )
    payload = {
        "fixture_dir": str(fixture_dir),
        "rows": rows,
        "summary": summary_rows,
        "selected_policies": list(selected_policies),
        "selected_p2_sources": list(selected_sources),
        "predictor_quality": {
            name: summarize_prediction_records(list(records.values()))
            for name, records in predictor_records.items()
        },
    }
    return payload


def render_prediction_replay_markdown(payload: dict[str, Any]) -> str:
    fixture_dir = payload["fixture_dir"]
    summary_rows = payload["summary"]
    md_lines = [
        "# Prediction Replay Summary",
        "",
        f"- fixture_dir: `{fixture_dir}`",
        "",
        "| Policy | P2 Source | Mean Makespan | Rel to zero | Gap to perfect | Fallback/B | Selected U | mean rel L1 | mean cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['policy_name']} | {row['p2_source']} | "
            f"{('-' if row['mean_makespan'] is None else f'{float(row['mean_makespan']):.0f}')} | "
            f"{('-' if row['relative_to_zero_hint'] is None else f'{100.0 * float(row['relative_to_zero_hint']):.2f}%')} | "
            f"{('-' if row['gap_to_perfect_trace'] is None else f'{100.0 * float(row['gap_to_perfect_trace']):.2f}%')} | "
            f"{100.0 * float(row['fallback_to_B_ratio']):.2f}% | "
            f"{100.0 * float(row['selected_U_ratio']):.2f}% | "
            f"{float(row['mean_prediction_relative_l1_error']):.4f} | "
            f"{float(row['mean_prediction_cosine_similarity']):.4f} |"
        )
    return "\n".join(md_lines) + "\n"


def main() -> None:
    args = _parse_args()
    payload = run_prediction_replay_suite(
        fixture_dir=Path(args.fixture_dir),
        policies=args.policies,
        p2_sources=args.p2_sources,
    )
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_summary_md).write_text(render_prediction_replay_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_summary": args.output_summary, "row_count": len(payload["rows"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
