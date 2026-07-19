#!/usr/bin/env python3
"""Trace P2 from matrix input to score/order/safe selection for main safe-U families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline._timeline_prediction_diagnosis_common import mean, top_edges, write_json, write_text
from experiments.offline.replay_fixture_policy_study import _build_problem
from rs.runtime.offline.prediction import rolling_predictor_records
from rs.scheduling.registry import resolve_policy
from rs.scheduling.policy_explain import explain_policy_decision
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_remote_bytes


POLICIES = ("RS_safe_barrier_criticality", "RS_safe_gated_greedy")
P2_SOURCES = (
    "zero_hint",
    "copy_current_dispatch",
    "fate_style_history",
    "fate_style_linear",
    "actual_trace_oracle",
    "perfect_trace_oracle",
    "amplified_actual_2x",
    "amplified_actual_4x",
    "shuffled_actual",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_fixtures(fixture_dir: Path) -> list[dict[str, Any]]:
    fixtures = []
    for path in sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1])):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        fixtures.append(payload)
    return fixtures


def _predictor_maps(fixture_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "fate_style_history": {record.layer_id: record for record in rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_history")},
        "fate_style_linear": {record.layer_id: record for record in rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_linear")},
    }


def _p2_matrix_for_source(
    fixture: dict[str, Any],
    *,
    source: str,
    predictors: dict[str, dict[str, Any]],
) -> tuple[tuple[int, ...], ...]:
    p0 = canonicalize_remote_matrix(fixture["p0_dispatch_matrix"])
    actual = canonicalize_remote_matrix(fixture.get("p2_next_dispatch_matrix", fixture["p2_next_dispatch_forecast_matrix"]))
    if source == "zero_hint":
        return tuple(tuple(0 for _ in row) for row in actual)
    if source == "copy_current_dispatch":
        return p0
    if source == "fate_style_history":
        record = predictors["fate_style_history"][str(fixture.get("metadata", {}).get("layer_id", ""))]
        return canonicalize_remote_matrix(record.predicted_matrix)
    if source == "fate_style_linear":
        record = predictors["fate_style_linear"][str(fixture.get("metadata", {}).get("layer_id", ""))]
        return canonicalize_remote_matrix(record.predicted_matrix)
    if source in {"actual_trace_oracle", "perfect_trace_oracle"}:
        return actual
    if source == "amplified_actual_2x":
        return canonicalize_remote_matrix(tuple(tuple(int(value) * 2 for value in row) for row in actual))
    if source == "amplified_actual_4x":
        return canonicalize_remote_matrix(tuple(tuple(int(value) * 4 for value in row) for row in actual))
    if source == "shuffled_actual":
        shuffled = [list(row[::-1]) for row in actual[::-1]]
        return canonicalize_remote_matrix(tuple(tuple(int(value) for value in row) for row in shuffled))
    raise ValueError(f"unsupported p2 source {source!r}")


def _to_build_problem_source(source: str) -> str:
    if source in {"actual_trace_oracle", "perfect_trace_oracle"}:
        return "actual_trace"
    if source in {"fate_style_history", "fate_style_linear"}:
        return source
    if source == "copy_current_dispatch":
        return "copy_current_dispatch"
    return "zero_hint"


def _kendall_tau(order_a: tuple[str, ...], order_b: tuple[str, ...]) -> float | None:
    common = [item for item in order_a if item in order_b]
    if len(common) < 2:
        return None
    index_b = {value: idx for idx, value in enumerate(order_b)}
    inversions = 0
    total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            total += 1
            if index_b[common[i]] > index_b[common[j]]:
                inversions += 1
    return 1.0 - (2.0 * inversions / total) if total else None


def _top_overlap(order_a: tuple[str, ...], order_b: tuple[str, ...], topk: int = 4) -> float:
    set_a = set(order_a[:topk])
    set_b = set(order_b[:topk])
    return float(len(set_a & set_b) / max(1, len(set_b)))


def _score_norm(score_rows: tuple[Any, ...]) -> float:
    return float(sum(abs(float(value)) for _edge, value in score_rows))


def _trace_component_norm(trace_steps: tuple[dict[str, Any], ...], key: str) -> float:
    total = 0.0
    for step in trace_steps:
        for item in step.get("ready", ()):
            total += abs(float(item.get(key, 0.0)))
    return total


def analyze_timeline(*, fixture_dir: Path, output_dir: Path) -> dict[str, Any]:
    fixtures = _load_fixtures(fixture_dir)
    predictors = _predictor_maps(fixture_dir)
    records: list[dict[str, Any]] = []
    zero_baseline: dict[tuple[str, int], dict[str, Any]] = {}
    barrier_diag_rows: list[dict[str, Any]] = []
    gated_diag_rows: list[dict[str, Any]] = []
    replay_summary: dict[str, dict[str, list[dict[str, Any]]]] = {policy: {} for policy in POLICIES}
    for fixture in fixtures:
        layer_id = int(fixture.get("metadata", {}).get("layer_id", 0))
        for source in P2_SOURCES:
            predicted_p2 = _p2_matrix_for_source(fixture, source=source, predictors=predictors)
            problem = _build_problem(
                fixture,
                mode="runtime_lookahead",
                p2_source=_to_build_problem_source(source),
                expert_compute_delay=0.0,
                predicted_p2_matrix=predicted_p2,
            )
            for policy_name in POLICIES:
                policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
                explain = explain_policy_decision(policy, problem, p2_hint=predicted_p2, p2_source=source)
                p2_norm = _trace_component_norm(explain.trace_steps, "prediction_component")
                total_norm = _trace_component_norm(explain.trace_steps, "score")
                non_p2_norm = max(0.0, total_norm - p2_norm)
                record = {
                    "layer_id": layer_id,
                    "window_id": layer_id,
                    "policy_name": policy_name,
                    "heuristic_family": "barrier_criticality_matching" if "barrier_criticality" in policy_name else "gated_greedy",
                    "p2_source": source,
                    "p0_remote_bytes": int(matrix_remote_bytes(problem.p0_dispatch_matrix)),
                    "p1_remote_bytes": int(matrix_remote_bytes(problem.p1_return_matrix)),
                    "p2_remote_bytes": int(matrix_remote_bytes(predicted_p2)),
                    "p2_top_edges": top_edges(predicted_p2),
                    "p2_score_norm": p2_norm,
                    "non_p2_score_norm": non_p2_norm,
                    "p2_influence_ratio": 0.0 if total_norm <= 1e-9 else p2_norm / total_norm,
                    "ready_edge_count": len(explain.ready_edges),
                    "blocked_edge_count": len(explain.blocked_edges),
                    "eligible_edge_count": len(explain.eligible_edges),
                    "selected_matching": list(explain.selected_matching),
                    "selected_order": list(explain.selected_order),
                    "paired_B_makespan": explain.paired_b_makespan,
                    "raw_U_makespan": explain.raw_u_makespan,
                    "safe_makespan": explain.modeled_makespan,
                    "safe_selected": explain.safe_selected,
                    "fallback_to_B": explain.fallback_to_b,
                    "fallback_reason": explain.fallback_reason,
                    "bottleneck_edges": list(explain.bottleneck_edges),
                    "critical_edges": list(explain.critical_edges),
                    "trace_steps": [dict(step) for step in explain.trace_steps],
                    "score_breakdown": {
                        "p0": list(explain.p0_score_by_edge),
                        "p1": list(explain.p1_score_by_edge),
                        "p2": list(explain.p2_score_by_edge),
                        "barrier": list(explain.barrier_score_by_edge),
                        "gate": list(explain.gate_score_by_edge),
                        "total": list(explain.total_score_by_edge),
                    },
                }
                if source == "zero_hint":
                    zero_baseline[(policy_name, layer_id)] = record
                else:
                    zero = zero_baseline.get((policy_name, layer_id))
                    if zero is not None:
                        record["order_diff_vs_zero"] = len(set(record["selected_order"]) ^ set(zero["selected_order"]))
                        record["matching_diff_vs_zero"] = len(set(record["selected_matching"]) ^ set(zero["selected_matching"]))
                        record["kendall_tau_vs_zero"] = _kendall_tau(tuple(record["selected_order"]), tuple(zero["selected_order"]))
                        record["top_k_order_overlap_vs_zero"] = _top_overlap(tuple(record["selected_order"]), tuple(zero["selected_order"]))
                        record["critical_edge_changed"] = list(record["critical_edges"][:1]) != list(zero["critical_edges"][:1])
                        record["bottleneck_edge_changed"] = list(record["bottleneck_edges"][:1]) != list(zero["bottleneck_edges"][:1])
                        record["delta_vs_zero_hint"] = float(record["safe_makespan"] - zero["safe_makespan"])
                    else:
                        record["order_diff_vs_zero"] = None
                        record["matching_diff_vs_zero"] = None
                        record["kendall_tau_vs_zero"] = None
                        record["top_k_order_overlap_vs_zero"] = None
                        record["critical_edge_changed"] = None
                        record["bottleneck_edge_changed"] = None
                        record["delta_vs_zero_hint"] = None
                record["delta_vs_paired_B"] = None if explain.paired_b_makespan is None else float(record["safe_makespan"] - explain.paired_b_makespan)
                records.append(record)
                replay_summary.setdefault(policy_name, {}).setdefault(source, []).append(record)
                if policy_name == "RS_safe_barrier_criticality" and source != "zero_hint":
                    barrier_diag_rows.append(record)
                if policy_name == "RS_safe_gated_greedy" and source != "zero_hint":
                    gated_diag_rows.append(record)

    barrier_diag = _classify_barrier(barrier_diag_rows)
    gated_diag = _classify_gated(gated_diag_rows)
    replay_comparison = _summarize_replay(replay_summary)
    prediction_track = _prediction_track_summary()
    timeline_payload = {
        "fixture_dir": str(fixture_dir),
        "records": records,
    }
    write_json(output_dir / "policy_decision_timeline.json", timeline_payload)
    write_text(output_dir / "policy_decision_timeline.md", _render_timeline_md(timeline_payload))
    write_json(output_dir / "barrier_criticality_p2_diagnosis.json", barrier_diag)
    write_text(output_dir / "barrier_criticality_p2_diagnosis.md", _render_diag_md("RS_safe_barrier_criticality", barrier_diag))
    write_json(output_dir / "gated_greedy_p2_diagnosis.json", gated_diag)
    write_text(output_dir / "gated_greedy_p2_diagnosis.md", _render_diag_md("RS_safe_gated_greedy", gated_diag))
    write_json(output_dir / "replay_comparison.json", replay_comparison)
    write_text(output_dir / "replay_comparison.md", _render_replay_md(replay_comparison))
    write_json(output_dir / "prediction_track_summary.json", prediction_track)
    write_text(output_dir / "prediction_track_summary.md", _render_prediction_track_md(prediction_track))
    return {
        "timeline": timeline_payload,
        "barrier": barrier_diag,
        "gated": gated_diag,
        "replay_comparison": replay_comparison,
        "prediction_track_summary": prediction_track,
    }


def _classify_barrier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classified = []
    for row in rows:
        label = "useful_order_change"
        if row["p2_score_norm"] <= 1e-9:
            label = "no_p2_score"
        elif row["p2_influence_ratio"] < 0.05:
            label = "p2_scale_dominated"
        elif not row["critical_edge_changed"] and not row["bottleneck_edge_changed"]:
            label = "p2_constant_shift"
        elif row["order_diff_vs_zero"] == 0:
            label = "p2_masked_by_eligibility"
        elif row["fallback_to_B"]:
            label = "safe_fallback_masks"
        elif row["delta_vs_zero_hint"] is not None and row["delta_vs_zero_hint"] > 0:
            label = "harmful_order_change"
        elif row["critical_edge_changed"] and not row["bottleneck_edge_changed"]:
            label = "order_changed_no_bottleneck_change"
        classified.append({"layer_id": row["layer_id"], "p2_source": row["p2_source"], "diagnosis": label, **row})
    return {
        "records": classified,
        "summary": {
            "diagnosis_counts": {label: sum(1 for row in classified if row["diagnosis"] == label) for label in sorted({row["diagnosis"] for row in classified})},
            "main_benefit_layers": sorted({row["layer_id"] for row in classified if (row.get("delta_vs_zero_hint") or 0.0) < 0.0}),
            "main_harm_layers": sorted({row["layer_id"] for row in classified if (row.get("delta_vs_zero_hint") or 0.0) > 0.0}),
            "fallback_layers": sorted({row["layer_id"] for row in classified if row.get("fallback_to_B")}),
        },
    }


def _classify_gated(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classified = []
    for row in rows:
        error_proxy = None
        if row["p2_source"] == "copy_current_dispatch":
            error_proxy = "copy_current_dispatch"
        label = "useful_order_change"
        if row["p2_influence_ratio"] > 0.35 and (row.get("delta_vs_zero_hint") or 0.0) > 0.0:
            label = "p2_scale_too_large"
        elif row["top_k_order_overlap_vs_zero"] is not None and row["top_k_order_overlap_vs_zero"] < 0.5 and (row.get("delta_vs_zero_hint") or 0.0) > 0.0:
            label = "top_edge_mismatch_harm"
        elif row.get("fallback_to_B"):
            label = "safe_fallback_saves_layer"
        elif row.get("delta_vs_zero_hint") is not None and row["delta_vs_zero_hint"] > 0.0:
            label = "harmful_order_change"
        classified.append({"layer_id": row["layer_id"], "p2_source": row["p2_source"], "diagnosis": label, "error_proxy": error_proxy, **row})
    return {
        "records": classified,
        "summary": {
            "diagnosis_counts": {label: sum(1 for row in classified if row["diagnosis"] == label) for label in sorted({row["diagnosis"] for row in classified})},
            "benefit_layers": sorted({row["layer_id"] for row in classified if (row.get("delta_vs_zero_hint") or 0.0) < 0.0}),
            "harm_layers": sorted({row["layer_id"] for row in classified if (row.get("delta_vs_zero_hint") or 0.0) > 0.0}),
            "fallback_layers": sorted({row["layer_id"] for row in classified if row.get("fallback_to_B")}),
        },
    }


def _summarize_replay(replay_summary: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"policies": {}}
    for policy_name, by_source in replay_summary.items():
        payload["policies"][policy_name] = {}
        zero_rows = by_source.get("zero_hint", [])
        zero_by_layer = {row["layer_id"]: row["safe_makespan"] for row in zero_rows}
        for source, rows in by_source.items():
            rel_zero = []
            for row in rows:
                baseline = zero_by_layer.get(row["layer_id"])
                if baseline:
                    rel_zero.append((float(row["safe_makespan"]) - float(baseline)) / float(baseline))
            payload["policies"][policy_name][source] = {
                "mean_makespan": mean([float(row["safe_makespan"]) for row in rows]),
                "relative_to_paired_B": mean([float(row["delta_vs_paired_B"]) / max(float(row["paired_B_makespan"]), 1.0) for row in rows if row.get("paired_B_makespan") is not None]),
                "relative_to_zero_hint": mean(rel_zero),
                "selected_U_ratio": mean([0.0 if row.get("fallback_to_B") else 1.0 for row in rows]),
                "fallback_to_B_ratio": mean([1.0 if row.get("fallback_to_B") else 0.0 for row in rows]),
                "benefit_layer_count": len({row["layer_id"] for row in rows if (row.get("delta_vs_zero_hint") or 0.0) < 0.0}),
                "harm_layer_count": len({row["layer_id"] for row in rows if (row.get("delta_vs_zero_hint") or 0.0) > 0.0}),
                "order_changed_layer_count": len({row["layer_id"] for row in rows if row.get("order_diff_vs_zero")}),
                "bottleneck_changed_layer_count": len({row["layer_id"] for row in rows if row.get("bottleneck_edge_changed")}),
            }
    return payload


def _prediction_track_summary() -> dict[str, Any]:
    return {
        "expert_prediction": {
            "status": "trace_ready_mapping_validated",
            "note": "corrected expert-to-traffic O1 is semantically validated; next step can evaluate expert-count predictors",
        },
        "traffic_prediction": {
            "status": "baseline_ready",
            "best_baseline": "fate_style_history",
            "note": "traffic-level predictor remains the most reliable current non-oracle baseline",
        },
        "policy_consumption": {
            "status": "primary_blocker",
            "note": "P2 consumption, not reconstruction, is the current blocker for contribution 2 gains",
        },
    }


def _render_timeline_md(payload: dict[str, Any]) -> str:
    lines = ["# Policy Decision Timeline", ""]
    lines.append("| layer | policy | p2_source | p2_influence_ratio | safe_makespan | delta_vs_zero_hint | fallback_to_B |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for row in payload["records"]:
        lines.append(
            f"| {row['layer_id']} | {row['policy_name']} | {row['p2_source']} | {row['p2_influence_ratio']:.4f} | "
            f"{row['safe_makespan']:.1f} | {row.get('delta_vs_zero_hint')} | {row.get('fallback_to_B')} |"
        )
    return "\n".join(lines) + "\n"


def _render_diag_md(policy_name: str, payload: dict[str, Any]) -> str:
    lines = [f"# {policy_name} P2 Diagnosis", ""]
    lines.append(f"- diagnosis_counts: `{payload['summary']['diagnosis_counts']}`")
    lines.append(f"- benefit_layers: `{payload['summary'].get('benefit_layers', payload['summary'].get('main_benefit_layers', []))}`")
    lines.append(f"- harm_layers: `{payload['summary'].get('harm_layers', payload['summary'].get('main_harm_layers', []))}`")
    lines.append(f"- fallback_layers: `{payload['summary'].get('fallback_layers', [])}`")
    return "\n".join(lines) + "\n"


def _render_replay_md(payload: dict[str, Any]) -> str:
    lines = ["# Replay Comparison", ""]
    for policy_name, by_source in payload["policies"].items():
        lines.append(f"## {policy_name}")
        lines.append("")
        lines.append("| p2_source | mean_makespan | relative_to_paired_B | relative_to_zero_hint | selected_U_ratio | fallback_to_B_ratio |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for source, row in by_source.items():
            lines.append(
                f"| {source} | {row['mean_makespan']} | {row['relative_to_paired_B']} | {row['relative_to_zero_hint']} | {row['selected_U_ratio']} | {row['fallback_to_B_ratio']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_prediction_track_md(payload: dict[str, Any]) -> str:
    lines = ["# Prediction Track Summary", ""]
    for key, row in payload.items():
        lines.append(f"## {key}")
        lines.append(f"- status: `{row['status']}`")
        lines.append(f"- note: {row['note']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    payload = analyze_timeline(fixture_dir=Path(args.fixture_dir), output_dir=Path(args.output_dir))
    print(json.dumps({"records": len(payload["timeline"]["records"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
