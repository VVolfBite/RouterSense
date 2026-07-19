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
from rs.runtime.online.megatron_ep.prediction import PredictedTrafficMatrix, compare_predicted_to_actual
from rs.runtime.online.megatron_ep.prediction.traffic_calibration import calibrate_traffic_matrix
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_remote_bytes
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
PREDICTOR_ROLLOUT_METADATA = {
    "fate_style_history": {
        "history_empty_fallback": "copy_current_dispatch",
        "used_current_sample_for_fit": False,
    },
    "fate_style_linear": {
        "history_empty_fallback": "copy_current_dispatch",
        "used_current_sample_for_fit": False,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--p2-sources", nargs="*", default=None)
    parser.add_argument(
        "--traffic-calibration",
        choices=("none", "oracle_total", "current_total", "history_layer_scale", "row_col_current", "row_col_history"),
        default="none",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _record_for_matrix(
    *,
    source_name: str,
    predictor_name: str,
    source_layer_id: str,
    target_layer_id: str,
    predicted_matrix: tuple[tuple[int, ...], ...],
    actual_matrix: tuple[tuple[int, ...], ...],
    evaluation_eligible: bool,
    oracle_prediction: bool,
) -> dict[str, Any]:
    predicted = PredictedTrafficMatrix(
        predictor_name=str(predictor_name),
        predictor_version="v1",
        source_layer_id=str(source_layer_id),
        predicted_layer_id=str(target_layer_id),
        matrix=canonicalize_remote_matrix(predicted_matrix),
        matrix_digest="",
        total_bytes=int(matrix_remote_bytes(predicted_matrix)),
        nonzero_edge_count=int(sum(1 for row in canonicalize_remote_matrix(predicted_matrix) for value in row if int(value) > 0)),
        confidence=1.0 if oracle_prediction else 0.5,
        is_oracle=bool(oracle_prediction),
        evaluation_eligible=bool(evaluation_eligible),
        created_at_phase="offline_replay",
    )
    audit = compare_predicted_to_actual(predicted, canonicalize_remote_matrix(actual_matrix))
    row_sum_error = float(
        sum(
            abs(sum(int(v) for v in predicted.matrix[src]) - sum(int(v) for v in canonicalize_remote_matrix(actual_matrix)[src]))
            for src in range(len(predicted.matrix))
        )
    )
    col_width = len(predicted.matrix[0]) if predicted.matrix else 0
    col_sum_error = float(
        sum(
            abs(
                sum(int(predicted.matrix[src][dst]) for src in range(len(predicted.matrix)))
                - sum(int(canonicalize_remote_matrix(actual_matrix)[src][dst]) for src in range(len(canonicalize_remote_matrix(actual_matrix))))
            )
            for dst in range(col_width)
        )
    )
    return {
        "p2_source": str(source_name),
        "predictor_name": str(predictor_name),
        "prediction_remote_l1_error": float(audit.absolute_l1_error),
        "prediction_relative_l1_error": float(audit.relative_l1_error),
        "prediction_cosine_similarity": float(audit.cosine_similarity),
        "topk_edge_overlap": float(audit.topk_edge_overlap),
        "nonzero_precision": float(audit.nonzero_edge_precision),
        "nonzero_recall": float(audit.nonzero_edge_recall),
        "row_sum_error": row_sum_error,
        "col_sum_error": col_sum_error,
        "forecast_remote_bytes": int(audit.predicted_remote_bytes),
        "actual_remote_bytes": int(audit.actual_remote_bytes),
        "evaluation_eligible": bool(evaluation_eligible),
        "oracle_prediction": bool(oracle_prediction),
    }


def run_prediction_replay_suite(
    *,
    fixture_dir: Path,
    policies: Iterable[str] | None = None,
    p2_sources: Iterable[str] | None = None,
    traffic_calibration: str = "none",
) -> dict[str, Any]:
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    selected_policies = tuple(policies or POLICIES)
    selected_sources = tuple(p2_sources or DEFAULT_SOURCES)
    expert_trace_available = bool(list(fixture_dir.glob("*expert_route_trace*.jsonl")) or list(fixture_dir.glob("*source_expert_counts*.jsonl")))
    expert_trace_reason = None if expert_trace_available else "expert_trace_unavailable_for_real_fixture"
    predictor_records = {
        "zero_hint": {record.layer_id: record for record in rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="zero_hint")},
        "copy_current_dispatch": {record.layer_id: record for record in rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="copy_current_dispatch")},
        "fate_style_history": {record.layer_id: record for record in rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_history")},
        "fate_style_linear": {record.layer_id: record for record in rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_linear")},
    }
    history_scale_by_source: dict[str, float] = {}
    previous_actual_by_source: dict[str, tuple[tuple[int, ...], ...]] = {}
    rows: list[dict[str, Any]] = []
    for fixture_path in fixture_paths:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        source_layer_id = str(fixture.get("metadata", {}).get("layer_id", ""))
        target_layer_id = str(fixture.get("metadata", {}).get("next_layer_id", ""))
        actual_target = canonicalize_remote_matrix(
            tuple(tuple(int(v) for v in row) for row in fixture.get("p2_next_dispatch_matrix", fixture["p2_next_dispatch_forecast_matrix"]))
        )
        current_dispatch = canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in fixture["p0_dispatch_matrix"]))
        for source in selected_sources:
            predicted_p2 = None
            calibration_audit = None
            predicted_raw = None
            evaluation_eligible = source not in {"perfect_trace", "actual_trace"}
            oracle_prediction = source in {"perfect_trace", "actual_trace"}
            predictor_name = source
            if source in predictor_records:
                record = predictor_records[source].get(source_layer_id)
                if record is not None:
                    predicted_raw = canonicalize_remote_matrix(record.predicted_matrix)
                    predictor_name = str(record.predictor_name)
            elif source == "perfect_trace":
                predicted_raw = actual_target
                predictor_name = "perfect_trace_oracle"
            elif source == "actual_trace":
                predicted_raw = actual_target
                predictor_name = "actual_trace_oracle"
            if predicted_raw is None:
                predicted_raw = canonicalize_remote_matrix(tuple(tuple(0 for _ in row) for row in actual_target))
            predicted_p2 = predicted_raw
            if traffic_calibration != "none":
                predicted_p2, calibration_audit = calibrate_traffic_matrix(
                    predicted_raw,
                    actual_matrix=actual_target,
                    current_dispatch_matrix=current_dispatch,
                    historical_reference_matrix=previous_actual_by_source.get(source),
                    historical_scale=history_scale_by_source.get(source),
                    mode=traffic_calibration,
                )
            prediction_metrics = _record_for_matrix(
                source_name=source,
                predictor_name=predictor_name,
                source_layer_id=source_layer_id,
                target_layer_id=target_layer_id,
                predicted_matrix=predicted_raw,
                actual_matrix=actual_target,
                evaluation_eligible=evaluation_eligible,
                oracle_prediction=oracle_prediction,
            )
            if calibration_audit is not None:
                calibrated_metrics = _record_for_matrix(
                    source_name=source,
                    predictor_name=f"{predictor_name}+{traffic_calibration}",
                    source_layer_id=source_layer_id,
                    target_layer_id=target_layer_id,
                    predicted_matrix=predicted_p2,
                    actual_matrix=actual_target,
                    evaluation_eligible=(traffic_calibration != "oracle_total" and evaluation_eligible),
                    oracle_prediction=oracle_prediction or traffic_calibration == "oracle_total",
                )
                prediction_metrics.update(
                    {
                        "traffic_calibration_mode": traffic_calibration,
                        "traffic_calibration_evaluation_eligible": traffic_calibration != "oracle_total",
                        "traffic_calibration_before_relative_l1": float(calibration_audit.before_relative_l1),
                        "traffic_calibration_after_relative_l1": float(calibration_audit.after_relative_l1),
                        "mean_traffic_error_before_calibration": float(calibration_audit.before_relative_l1),
                        "mean_traffic_error_after_calibration": float(calibration_audit.after_relative_l1),
                        "median_traffic_error_before_calibration": float(calibration_audit.before_relative_l1),
                        "median_traffic_error_after_calibration": float(calibration_audit.after_relative_l1),
                        "predictor_quality_raw": {
                            "relative_l1_error": float(prediction_metrics["prediction_relative_l1_error"]),
                            "cosine_similarity": float(prediction_metrics["prediction_cosine_similarity"]),
                        },
                        "predictor_quality_calibrated": {
                            "relative_l1_error": float(calibrated_metrics["prediction_relative_l1_error"]),
                            "cosine_similarity": float(calibrated_metrics["prediction_cosine_similarity"]),
                        },
                    }
                )
            else:
                prediction_metrics.update(
                    {
                        "traffic_calibration_mode": traffic_calibration,
                        "traffic_calibration_evaluation_eligible": traffic_calibration != "oracle_total",
                        "traffic_calibration_before_relative_l1": None,
                        "traffic_calibration_after_relative_l1": None,
                        "mean_traffic_error_before_calibration": None,
                        "mean_traffic_error_after_calibration": None,
                        "median_traffic_error_before_calibration": None,
                        "median_traffic_error_after_calibration": None,
                        "predictor_quality_raw": {
                            "relative_l1_error": float(prediction_metrics["prediction_relative_l1_error"]),
                            "cosine_similarity": float(prediction_metrics["prediction_cosine_similarity"]),
                        },
                        "predictor_quality_calibrated": None,
                    }
                )
            prediction_metrics.update(
                {
                    "expert_prediction_available": False,
                    "expert_count_relative_l1_error": None,
                    "expert_topk_overlap": None,
                    "expert_to_traffic_reconstruction_error": None,
                    "traffic_error_from_predicted_experts": None,
                    "gate_replay_available": False,
                    "expert_trace_available": expert_trace_available,
                    "expert_trace_unavailable_reason": expert_trace_reason,
                }
            )
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
                        "topk_edge_overlap": prediction_metrics["topk_edge_overlap"],
                        "nonzero_precision": prediction_metrics["nonzero_precision"],
                        "nonzero_recall": prediction_metrics["nonzero_recall"],
                        "row_sum_error": prediction_metrics["row_sum_error"],
                        "col_sum_error": prediction_metrics["col_sum_error"],
                        "forecast_remote_bytes": prediction_metrics["forecast_remote_bytes"],
                        "actual_remote_bytes": prediction_metrics["actual_remote_bytes"],
                        "safe_makespan": float(audit.get("makespan", plan.diagnostics.get("makespan", 0.0))),
                        "fallback_to_B": bool(plan.diagnostics.get("fallback_to_paired_b", False)),
                        "selected_policy": str(plan.diagnostics.get("selected_policy", plan.policy_name)),
                        "evaluation_eligible": bool(plan.diagnostics.get("evaluation_eligible", True)),
                        "oracle_prediction": bool(prediction_metrics["oracle_prediction"]),
                        "expert_trace_available": prediction_metrics["expert_trace_available"],
                        "expert_prediction_available": prediction_metrics["expert_prediction_available"],
                        "expert_count_relative_l1_error": prediction_metrics["expert_count_relative_l1_error"],
                        "expert_topk_overlap": prediction_metrics["expert_topk_overlap"],
                        "expert_to_traffic_reconstruction_error": prediction_metrics["expert_to_traffic_reconstruction_error"],
                        "traffic_error_from_predicted_experts": prediction_metrics["traffic_error_from_predicted_experts"],
                        "gate_replay_available": prediction_metrics["gate_replay_available"],
                        "expert_trace_unavailable_reason": prediction_metrics["expert_trace_unavailable_reason"],
                        "traffic_calibration_mode": prediction_metrics.get("traffic_calibration_mode"),
                        "traffic_calibration_evaluation_eligible": prediction_metrics.get("traffic_calibration_evaluation_eligible"),
                        "traffic_calibration_before_relative_l1": prediction_metrics.get("traffic_calibration_before_relative_l1"),
                        "traffic_calibration_after_relative_l1": prediction_metrics.get("traffic_calibration_after_relative_l1"),
                        "mean_traffic_error_before_calibration": prediction_metrics.get("mean_traffic_error_before_calibration"),
                        "mean_traffic_error_after_calibration": prediction_metrics.get("mean_traffic_error_after_calibration"),
                        "median_traffic_error_before_calibration": prediction_metrics.get("median_traffic_error_before_calibration"),
                        "median_traffic_error_after_calibration": prediction_metrics.get("median_traffic_error_after_calibration"),
                        "predictor_quality_raw": prediction_metrics.get("predictor_quality_raw"),
                        "predictor_quality_calibrated": prediction_metrics.get("predictor_quality_calibrated"),
                    }
                )
            rows.extend(policy_results)
            predicted_total = max(1, int(matrix_remote_bytes(predicted_raw)))
            actual_total = max(1, int(matrix_remote_bytes(actual_target)))
            history_scale_by_source[source] = float(actual_total) / float(predicted_total)
            previous_actual_by_source[source] = actual_target
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
                    "mean_topk_edge_overlap": statistics.mean([float(row["topk_edge_overlap"]) for row in source_rows]) if source_rows else 0.0,
                    "mean_nonzero_precision": statistics.mean([float(row["nonzero_precision"]) for row in source_rows]) if source_rows else 0.0,
                    "mean_nonzero_recall": statistics.mean([float(row["nonzero_recall"]) for row in source_rows]) if source_rows else 0.0,
                    "mean_row_sum_error": statistics.mean([float(row["row_sum_error"]) for row in source_rows]) if source_rows else 0.0,
                    "mean_col_sum_error": statistics.mean([float(row["col_sum_error"]) for row in source_rows]) if source_rows else 0.0,
                    "forecast_remote_bytes": _mean([float(row["forecast_remote_bytes"]) for row in source_rows]),
                    "actual_remote_bytes": _mean([float(row["actual_remote_bytes"]) for row in source_rows]),
                    "expert_trace_available": bool(source_rows[0]["expert_trace_available"]) if source_rows else False,
                    "expert_prediction_available": bool(source_rows[0]["expert_prediction_available"]) if source_rows else False,
                    "expert_count_relative_l1_error": None if not source_rows else source_rows[0]["expert_count_relative_l1_error"],
                    "expert_topk_overlap": None if not source_rows else source_rows[0]["expert_topk_overlap"],
                    "expert_to_traffic_reconstruction_error": None if not source_rows else source_rows[0]["expert_to_traffic_reconstruction_error"],
                    "traffic_error_from_predicted_experts": None if not source_rows else source_rows[0]["traffic_error_from_predicted_experts"],
                    "gate_replay_available": bool(source_rows[0]["gate_replay_available"]) if source_rows else False,
                    "expert_trace_unavailable_reason": None if not source_rows else source_rows[0]["expert_trace_unavailable_reason"],
                    "traffic_calibration_mode": None if not source_rows else source_rows[0].get("traffic_calibration_mode"),
                    "mean_traffic_error_before_calibration": _mean([float(row["traffic_calibration_before_relative_l1"]) for row in source_rows if row.get("traffic_calibration_before_relative_l1") is not None]),
                    "mean_traffic_error_after_calibration": _mean([float(row["traffic_calibration_after_relative_l1"]) for row in source_rows if row.get("traffic_calibration_after_relative_l1") is not None]),
                    "median_traffic_error_before_calibration": _median([float(row["traffic_calibration_before_relative_l1"]) for row in source_rows if row.get("traffic_calibration_before_relative_l1") is not None]),
                    "median_traffic_error_after_calibration": _median([float(row["traffic_calibration_after_relative_l1"]) for row in source_rows if row.get("traffic_calibration_after_relative_l1") is not None]),
                    "evaluation_eligible": all(bool(row["evaluation_eligible"]) for row in source_rows) if source_rows else False,
                    "oracle_prediction": bool(source_rows[0]["oracle_prediction"]) if source_rows else False,
                    "predictor_quality_raw": None
                    if not source_rows
                    else {
                        "mean_relative_l1_error": statistics.mean([float(row["prediction_relative_l1_error"]) for row in source_rows]),
                        "mean_cosine_similarity": statistics.mean([float(row["prediction_cosine_similarity"]) for row in source_rows]),
                    },
                    "predictor_quality_calibrated": None
                    if not source_rows or all(row.get("traffic_calibration_after_relative_l1") is None for row in source_rows)
                    else {
                        "mean_relative_l1_error": statistics.mean(
                            [float(row["traffic_calibration_after_relative_l1"]) for row in source_rows if row.get("traffic_calibration_after_relative_l1") is not None]
                        ),
                        "mean_cosine_similarity": statistics.mean(
                            [float(row["prediction_cosine_similarity"]) for row in source_rows]
                        ),
                    },
                }
            )
    payload = {
        "fixture_dir": str(fixture_dir),
        "rows": rows,
        "summary": summary_rows,
        "selected_policies": list(selected_policies),
        "selected_p2_sources": list(selected_sources),
        "traffic_calibration": traffic_calibration,
        "expert_trace_available": expert_trace_available,
        "expert_trace_unavailable_reason": expert_trace_reason,
        "predictor_quality_raw": {
            name: {
                **summarize_prediction_records(list(records.values())),
                **PREDICTOR_ROLLOUT_METADATA.get(name, {}),
            }
            for name, records in predictor_records.items()
        },
        "predictor_quality_calibrated": {
            row["p2_source"]: row["predictor_quality_calibrated"]
            for row in summary_rows
            if row.get("predictor_quality_calibrated") is not None
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
        f"- traffic_calibration: `{payload.get('traffic_calibration', 'none')}`",
        "- note: `oracle_total` is an oracle diagnostic and is not evaluation-eligible for online claims.",
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
        traffic_calibration=args.traffic_calibration,
    )
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_summary_md).write_text(render_prediction_replay_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_summary": args.output_summary, "row_count": len(payload["rows"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
