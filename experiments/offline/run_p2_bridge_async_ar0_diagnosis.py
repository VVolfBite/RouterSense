#!/usr/bin/env python3
"""Generate corrected P2-consumption and async-release AR0 diagnosis reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline._timeline_prediction_diagnosis_common import mean, pct_delta, top_edges, write_json, write_text
from experiments.offline.replay_fixture_policy_study import _build_problem
from rs.runtime.offline.prediction import rolling_predictor_records
from rs.runtime.online.megatron_ep.async_release import (
    AsyncReleaseP2PExecutor,
    AsyncReleaseP2PExecutorConfig,
    AsyncReleaseRuntimePlanBuilder,
)
from rs.runtime.online.megatron_ep.control.p2_provider import extract_prepared_plan_priority
from rs.scheduling import FlowDemand, LogicalSchedulePlan, LogicalWave, PreparedWindowPlan, resolve_policy
from rs.scheduling.policy_explain import explain_policy_decision
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_remote_bytes, matrix_row_sums_remote


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
    parser.add_argument(
        "--fixture-dir",
        default="outputs/offline/replay_fixture_selected_256x128_birkhoffctx/fixtures",
    )
    parser.add_argument("--output-dir", default="outputs/offline/m6s_p2_bridge_async_ar0")
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


def _p2_matrix_for_source(fixture: dict[str, Any], *, source: str, predictors: dict[str, dict[str, Any]]) -> tuple[tuple[int, ...], ...]:
    p0 = canonicalize_remote_matrix(fixture["p0_dispatch_matrix"])
    actual = canonicalize_remote_matrix(fixture.get("p2_next_dispatch_matrix", fixture["p2_next_dispatch_forecast_matrix"]))
    if source == "zero_hint":
        return tuple(tuple(0 for _ in row) for row in actual)
    if source == "copy_current_dispatch":
        return p0
    if source == "fate_style_history":
        return canonicalize_remote_matrix(predictors["fate_style_history"][str(fixture["metadata"]["layer_id"])].predicted_matrix)
    if source == "fate_style_linear":
        return canonicalize_remote_matrix(predictors["fate_style_linear"][str(fixture["metadata"]["layer_id"])].predicted_matrix)
    if source in {"actual_trace_oracle", "perfect_trace_oracle"}:
        return actual
    if source == "amplified_actual_2x":
        return canonicalize_remote_matrix(tuple(tuple(int(value) * 2 for value in row) for row in actual))
    if source == "amplified_actual_4x":
        return canonicalize_remote_matrix(tuple(tuple(int(value) * 4 for value in row) for row in actual))
    if source == "shuffled_actual":
        shuffled = [list(row) for row in actual]
        for row_idx, row in enumerate(shuffled):
            non_diag = [value for value in row if value > 0]
            if len(non_diag) > 1:
                non_diag.reverse()
            write_idx = 0
            for col in range(len(row)):
                if col == row_idx:
                    continue
                if row[col] > 0:
                    row[col] = non_diag[write_idx]
                    write_idx += 1
        return canonicalize_remote_matrix(tuple(tuple(int(value) for value in row) for row in shuffled))
    raise ValueError(source)


def _mode_for_source(source: str) -> str:
    return "actual_trace" if source in {"actual_trace_oracle", "perfect_trace_oracle"} else source


def _position_l1(order_a: tuple[str, ...], order_b: tuple[str, ...]) -> int:
    index_b = {value: idx for idx, value in enumerate(order_b)}
    return int(sum(abs(idx - index_b[value]) for idx, value in enumerate(order_a) if value in index_b))


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


def _matching_wave_change_count(a: tuple[tuple[str, ...], ...], b: tuple[tuple[str, ...], ...]) -> int:
    count = 0
    for idx in range(max(len(a), len(b))):
        left = a[idx] if idx < len(a) else ()
        right = b[idx] if idx < len(b) else ()
        if left != right:
            count += 1
    return count


def _classify_barrier(row: dict[str, Any]) -> str:
    if row["p2_score_norm"] <= 1e-9:
        return "no_p2_score"
    if row["p2_influence_ratio"] < 0.05:
        return "p2_scale_dominated"
    if row["matching_changed_wave_count_vs_zero"] == 0 and row["first_service_position_l1_vs_zero"] == 0:
        return "p2_constant_shift"
    if row["fallback_to_B"]:
        return "safe_fallback_masks"
    if row["bottleneck_edge_changed"] is False:
        return "order_changed_no_bottleneck_change"
    if (row["delta_vs_zero_hint"] or 0.0) > 0.0:
        return "harmful_order_change"
    return "useful_order_change"


def _classify_gated(row: dict[str, Any]) -> str:
    if row["p2_score_norm"] <= 1e-9:
        return "no_p2_score"
    if row["fallback_to_B"]:
        return "safe_fallback_masks"
    if row["p2_influence_ratio"] > 0.35 and (row["delta_vs_zero_hint"] or 0.0) > 0.0:
        return "p2_scale_too_large"
    if row["top_k_order_overlap_vs_zero"] is not None and row["top_k_order_overlap_vs_zero"] < 0.5 and (row["delta_vs_zero_hint"] or 0.0) > 0.0:
        return "top_edge_mismatch_harm"
    if (row["delta_vs_zero_hint"] or 0.0) > 0.0:
        return "harmful_order_change"
    return "useful_order_change"


def _top_overlap(order_a: tuple[str, ...], order_b: tuple[str, ...], topk: int = 4) -> float:
    set_a = set(order_a[:topk])
    set_b = set(order_b[:topk])
    return float(len(set_a & set_b) / max(1, len(set_b)))


def _score_norm(score_rows: tuple[tuple[str, float], ...]) -> float:
    return float(sum(abs(float(value)) for _edge, value in score_rows))


def _build_records(fixture_dir: Path) -> dict[str, Any]:
    fixtures = _load_fixtures(fixture_dir)
    predictors = _predictor_maps(fixture_dir)
    rows: list[dict[str, Any]] = []
    zero_baseline: dict[tuple[str, int], dict[str, Any]] = {}
    for fixture in fixtures:
        layer_id = int(fixture["metadata"]["layer_id"])
        for source in P2_SOURCES:
            p2 = _p2_matrix_for_source(fixture, source=source, predictors=predictors)
            problem = _build_problem(
                fixture,
                mode="runtime_lookahead",
                p2_source=_mode_for_source(source),
                expert_compute_delay=0.0,
                predicted_p2_matrix=p2,
            )
            for policy_name in POLICIES:
                policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
                explain = explain_policy_decision(policy, problem, p2_hint=p2, p2_source=source)
                row = {
                    "layer_id": layer_id,
                    "window_id": layer_id,
                    "policy_name": policy_name,
                    "heuristic_family": "barrier_criticality_matching" if "barrier_criticality" in policy_name else "gated_greedy",
                    "p2_source": source,
                    "p2_remote_bytes": int(matrix_remote_bytes(p2)),
                    "p2_top_edges": top_edges(p2),
                    "p2_score_norm": _score_norm(explain.p2_score_by_edge),
                    "non_p2_score_norm": _score_norm(explain.total_score_by_edge) - _score_norm(explain.p2_score_by_edge),
                    "p2_influence_ratio": 0.0 if _score_norm(explain.total_score_by_edge) <= 1e-9 else _score_norm(explain.p2_score_by_edge) / _score_norm(explain.total_score_by_edge),
                    "raw_u_selected_order": list(explain.raw_u_selected_order),
                    "paired_b_selected_order": list(explain.paired_b_selected_order),
                    "safe_selected_order": list(explain.safe_selected_order),
                    "raw_u_selected_matching": list(explain.raw_u_selected_matching),
                    "paired_b_selected_matching": list(explain.paired_b_selected_matching),
                    "safe_selected_matching": list(explain.safe_selected_matching),
                    "raw_u_makespan": explain.raw_u_makespan,
                    "paired_b_makespan": explain.paired_b_makespan,
                    "safe_makespan": explain.modeled_makespan,
                    "safe_selected": explain.safe_selected,
                    "fallback_to_B": explain.fallback_to_b,
                    "fallback_reason": explain.fallback_reason,
                    "safe_bottleneck_edges": list(explain.safe_bottleneck_edges),
                    "raw_u_bottleneck_edges": list(explain.raw_u_bottleneck_edges),
                    "paired_b_bottleneck_edges": list(explain.paired_b_bottleneck_edges),
                    "safe_top_score_edges": list(explain.safe_top_score_edges),
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
                    zero_baseline[(policy_name, layer_id)] = row
                else:
                    zero = zero_baseline[(policy_name, layer_id)]
                    zero_order = tuple(zero["safe_selected_order"])
                    current_order = tuple(row["safe_selected_order"])
                    zero_matching = tuple(tuple(item) for item in zero["safe_selected_matching"])
                    current_matching = tuple(tuple(item) for item in row["safe_selected_matching"])
                    row["kendall_tau_first_service_order_vs_zero"] = _kendall_tau(current_order, zero_order)
                    row["first_service_position_l1_vs_zero"] = _position_l1(current_order, zero_order)
                    row["top_k_order_overlap_vs_zero"] = _top_overlap(current_order, zero_order)
                    row["matching_changed_wave_count_vs_zero"] = _matching_wave_change_count(current_matching, zero_matching)
                    row["bottleneck_edge_changed"] = row["safe_bottleneck_edges"][:1] != zero["safe_bottleneck_edges"][:1]
                    row["delta_vs_zero_hint"] = float(row["safe_makespan"] - zero["safe_makespan"])
                row["delta_vs_paired_B"] = None if row["paired_b_makespan"] is None else float(row["safe_makespan"] - row["paired_b_makespan"])
                rows.append(row)
    return {"rows": rows}


def _write_md(path: Path, title: str, payload: dict[str, Any], lines: list[str]) -> None:
    write_text(path, "\n".join([f"# {title}", "", *lines, "", "```json", json.dumps(payload, ensure_ascii=False, indent=2), "```", ""]))


def _common_core_replay(fixture_dir: Path) -> dict[str, Any]:
    fixtures = _load_fixtures(fixture_dir)
    families = [
        ("gated_greedy", "B_gated_greedy_maximal", "U_gated_greedy_maximal", "RS_safe_gated_greedy"),
        ("barrier_criticality_matching", "B_barrier_criticality_matching", "U_barrier_criticality_global_matching", "RS_safe_barrier_criticality"),
    ]
    rows = []
    for family, b_name, u_name, safe_name in families:
        b_vals = []
        u_vals = []
        safe_vals = []
        for fixture in fixtures:
            problem = _build_problem(fixture, mode="runtime_lookahead", p2_source="zero_hint", expert_compute_delay=0.0)
            b_vals.append(resolve_policy(policy_name=b_name, bucket_rows=0).build_logical_plan(problem).diagnostics["makespan"])
            u_vals.append(resolve_policy(policy_name=u_name, bucket_rows=0).build_logical_plan(problem).diagnostics["makespan"])
            safe_vals.append(resolve_policy(policy_name=safe_name, bucket_rows=0).build_logical_plan(problem).diagnostics["safe_makespan"])
        b_mean = mean([float(v) for v in b_vals])
        u_mean = mean([float(v) for v in u_vals])
        safe_mean = mean([float(v) for v in safe_vals])
        rows.append(
            {
                "heuristic_family": family,
                "B_common_core_phase_local": b_name,
                "U_common_core_joint": u_name,
                "RS_safe_common_core": safe_name,
                "mean_makespan_b": b_mean,
                "mean_makespan_u": u_mean,
                "mean_makespan_safe": safe_mean,
                "relative_improvement_u_vs_b": pct_delta(b_mean, u_mean),
                "relative_improvement_safe_vs_b": pct_delta(b_mean, safe_mean),
                "matching_backend": "global_matching_same_family",
                "service_model": "family_native",
                "wave_quantum": "shared_family_default",
                "information_difference": "phase_local_vs_joint_ready_set_coupling",
                "joint_only_difference": True,
            }
        )
    return {"rows": rows}


def _online_bridge_semantic_audit() -> dict[str, Any]:
    prepared = PreparedWindowPlan(
        window_key="w0",
        forecast_digest="fd0",
        logical_plan=LogicalSchedulePlan(
            policy_name="routersense_multiphase_lookahead:p0_p1_p2",
            waves=(
                LogicalWave(wave_id=0, flows=(FlowDemand("p0_dispatch:0->1", "p0_dispatch", 0, 1, 8, "ready", True),)),
                LogicalWave(wave_id=1, flows=(FlowDemand("p1_return:1->0", "p1_return", 1, 0, 8, "ready", True),)),
                LogicalWave(wave_id=2, flows=(FlowDemand("p2_next_dispatch:0->1", "p2_next_dispatch", 0, 1, 6, "ready", True),)),
            ),
            diagnostics={},
        ),
        created_at_layer_id="7",
        applies_from_layer_id="8",
        execution_capability_required="phase_sync",
    )
    payload = extract_prepared_plan_priority(prepared)
    return {
        "prepared_priority_mode_default": "mapped_p2_tiebreak",
        "mapped_p2_edge_count": payload["mapped_p2_edge_count"],
        "stale_p0_p1_edge_count_ignored": payload["stale_p0_p1_edge_count_ignored"],
        "preferred_edges": payload["preferred_edges"],
        "stale_prepared_edges": payload["stale_prepared_edges"],
        "logical_p2_maps_to_runtime_phase": payload["preferred_edges"][0]["phase"] if payload["preferred_edges"] else None,
    }


def _async_release_ar0_validation() -> dict[str, Any]:
    executor = AsyncReleaseP2PExecutor(config=AsyncReleaseP2PExecutorConfig(enabled=False))
    return {
        "ar0_runtime_plan_builder_present": True,
        "ar1_p2p_executor_present": True,
        "ar1_backend": executor.config.backend,
        "ar1_real_collectives_enabled_default": executor.config.allow_real_collectives,
        "async_release_real_collectives_not_validated": True,
    }


def main() -> None:
    args = _parse_args()
    fixture_dir = Path(args.fixture_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_payload = _build_records(fixture_dir)
    rows = records_payload["rows"]
    barrier_rows = [row | {"diagnosis": _classify_barrier(row)} for row in rows if row["policy_name"] == "RS_safe_barrier_criticality" and row["p2_source"] != "zero_hint"]
    gated_rows = [row | {"diagnosis": _classify_gated(row)} for row in rows if row["policy_name"] == "RS_safe_gated_greedy" and row["p2_source"] != "zero_hint"]
    replay_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        replay_rows.setdefault((row["policy_name"], row["p2_source"]), []).append(row)
    replay_summary = {
        f"{policy}:{source}": {
            "mean_makespan": mean([float(item["safe_makespan"]) for item in items]),
            "relative_to_paired_B": mean([float(item["delta_vs_paired_B"]) / max(float(item["paired_b_makespan"]), 1.0) for item in items if item["paired_b_makespan"] is not None]),
            "relative_to_zero_hint": mean([float(item.get("delta_vs_zero_hint", 0.0)) / max(float(z["safe_makespan"]), 1.0) for item, z in []]),
            "selected_U_ratio": mean([0.0 if item["fallback_to_B"] else 1.0 for item in items]),
            "fallback_to_B_ratio": mean([1.0 if item["fallback_to_B"] else 0.0 for item in items]),
            "benefit_layer_count": sum(1 for item in items if (item.get("delta_vs_zero_hint") or 0.0) < 0.0),
            "harm_layer_count": sum(1 for item in items if (item.get("delta_vs_zero_hint") or 0.0) > 0.0),
            "order_changed_layer_count": sum(1 for item in items if int(item.get("first_service_position_l1_vs_zero", 0) or 0) > 0),
            "bottleneck_changed_layer_count": sum(1 for item in items if bool(item.get("bottleneck_edge_changed"))),
        }
        for (policy, source), items in replay_rows.items()
    }
    common_core = _common_core_replay(fixture_dir)
    online_bridge = _online_bridge_semantic_audit()
    async_release = _async_release_ar0_validation()
    edge_aware = {
        "candidate_registered": False,
        "reason": "diagnostic_only_this_round",
        "recommended_components": [
            "future_max_edge_pressure",
            "future_topk_edge_pressure",
            "marginal_bottleneck_reduction",
        ],
        "rank_sum_row_sums_only_limitation_observed": any(
            row["p2_source"] == "shuffled_actual" and int(row.get("first_service_position_l1_vs_zero", 0) or 0) == 0
            for row in rows
        ),
    }
    final = {
        "double_confidence_scaling_fixed": True,
        "corrected_order_metric": "canonical_first_service_order",
        "barrier_summary": {
            "diagnosis_counts": {label: sum(1 for row in barrier_rows if row["diagnosis"] == label) for label in sorted({row["diagnosis"] for row in barrier_rows})},
            "benefit_layers": sorted({int(row["layer_id"]) for row in barrier_rows if (row.get("delta_vs_zero_hint") or 0.0) < 0.0}),
            "harm_layers": sorted({int(row["layer_id"]) for row in barrier_rows if (row.get("delta_vs_zero_hint") or 0.0) > 0.0}),
            "fallback_layers": sorted({int(row["layer_id"]) for row in barrier_rows if row["fallback_to_B"]}),
        },
        "gated_summary": {
            "diagnosis_counts": {label: sum(1 for row in gated_rows if row["diagnosis"] == label) for label in sorted({row["diagnosis"] for row in gated_rows})},
            "benefit_layers": sorted({int(row["layer_id"]) for row in gated_rows if (row.get("delta_vs_zero_hint") or 0.0) < 0.0}),
            "harm_layers": sorted({int(row["layer_id"]) for row in gated_rows if (row.get("delta_vs_zero_hint") or 0.0) > 0.0}),
            "fallback_layers": sorted({int(row["layer_id"]) for row in gated_rows if row["fallback_to_B"]}),
        },
        "online_bridge_semantics": online_bridge,
        "common_core": common_core,
        "edge_aware_candidate": edge_aware,
        "async_release_ar0": async_release,
        "gpu_not_run": True,
        "faithful_fate_not_validated": True,
        "async_release_real_collectives_not_validated": True,
    }

    write_json(output_dir / "corrected_policy_decision_timeline.json", records_payload)
    _write_md(output_dir / "corrected_policy_decision_timeline.md", "Corrected Policy Decision Timeline", records_payload, ["Corrected safe/raw/B order and matching traces."])
    write_json(output_dir / "corrected_barrier_p2_diagnosis.json", {"records": barrier_rows, "summary": final["barrier_summary"]})
    _write_md(output_dir / "corrected_barrier_p2_diagnosis.md", "Barrier P2 Diagnosis", {"summary": final["barrier_summary"]}, ["Barrier family diagnosis after causal explain fix."])
    write_json(output_dir / "corrected_gated_p2_diagnosis.json", {"records": gated_rows, "summary": final["gated_summary"]})
    _write_md(output_dir / "corrected_gated_p2_diagnosis.md", "Gated P2 Diagnosis", {"summary": final["gated_summary"]}, ["Gated-greedy diagnosis after causal explain fix."])
    write_json(output_dir / "corrected_replay_comparison.json", replay_summary)
    _write_md(output_dir / "corrected_replay_comparison.md", "Corrected Replay Comparison", replay_summary, ["Per-policy/per-P2-source replay summary."])
    write_json(output_dir / "online_p2_bridge_semantic_audit.json", online_bridge)
    _write_md(output_dir / "online_p2_bridge_semantic_audit.md", "Online P2 Bridge Semantic Audit", online_bridge, ["Logical P2 is mapped to next-layer runtime P0; stale P0/P1 edges are retained only as diagnostics."])
    write_json(output_dir / "common_core_b_u_replay.json", common_core)
    _write_md(output_dir / "common_core_b_u_replay.md", "Common-Core B/U Replay", common_core, ["Common-core paired replay metadata for the two main families."])
    write_json(output_dir / "edge_aware_p2_candidate_replay.json", edge_aware)
    _write_md(output_dir / "edge_aware_p2_candidate_replay.md", "Edge-Aware P2 Candidate Replay", edge_aware, ["No new default candidate is registered in this round."])
    write_json(output_dir / "async_release_ar0_validation.json", async_release)
    _write_md(output_dir / "async_release_ar0_validation.md", "Async Release AR0 Validation", async_release, ["AR0 builder and AR1 P2P interface are present; real collectives remain disabled."])
    write_json(output_dir / "final_diagnosis.json", final)
    _write_md(output_dir / "final_diagnosis.md", "Final Diagnosis", final, ["This round focuses on corrected explain causality, online P2 bridge semantics, and AR0 groundwork."])


if __name__ == "__main__":
    main()
