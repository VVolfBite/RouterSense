#!/usr/bin/env python3
"""Offline replay for P2-signal sensitivity on safe-U policies."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any
from rs.runtime.offline.policy_study import build_replay_problem
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling import resolve_policy
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_remote_bytes


POLICIES = ("RS_safe_barrier_criticality", "RS_safe_gated_greedy")


def _scale(matrix: tuple[tuple[int, ...], ...], factor: int) -> tuple[tuple[int, ...], ...]:
    return canonicalize_remote_matrix(tuple(tuple(int(value) * int(factor) for value in row) for row in matrix))


def _shuffle(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    width = len(matrix[0]) if matrix else 0
    return canonicalize_remote_matrix(
        tuple(
            tuple(int(matrix[(src + 1) % len(matrix)][(dst + 1) % width]) for dst in range(width))
            for src in range(len(matrix))
        )
    )


def _variant_matrix(
    *,
    fixture: dict[str, Any],
    variant: str,
) -> tuple[tuple[int, ...], ...]:
    actual = canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in fixture["p2_next_dispatch_matrix"]))
    current = canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in fixture["p0_dispatch_matrix"]))
    if variant == "zero_hint":
        return canonicalize_remote_matrix(tuple(tuple(0 for _ in row) for row in actual))
    if variant == "copy_current_dispatch":
        return current
    if variant in {"actual_trace", "perfect_trace"}:
        return actual
    if variant == "amplified_actual_2x":
        return _scale(actual, 2)
    if variant == "amplified_actual_4x":
        return _scale(actual, 4)
    if variant == "shuffled_actual":
        return _shuffle(actual)
    raise ValueError(f"unsupported p2 variant {variant!r}")


def run_p2_sensitivity_replay(*, fixture_dir: Path) -> dict[str, Any]:
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    variants = (
        "zero_hint",
        "copy_current_dispatch",
        "actual_trace",
        "perfect_trace",
        "amplified_actual_2x",
        "amplified_actual_4x",
        "shuffled_actual",
    )
    rows: list[dict[str, Any]] = []
    for fixture_path in fixture_paths:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        layer_id = str(fixture.get("metadata", {}).get("layer_id", ""))
        zero_baseline: dict[str, float] = {}
        for variant in variants:
            predicted = _variant_matrix(fixture=fixture, variant=variant)
            problem = build_replay_problem(
                fixture,
                mode="runtime_lookahead",
                p2_source="copy_current_dispatch" if variant not in {"actual_trace", "perfect_trace"} else "actual_trace",
                expert_compute_delay=0.0,
                predicted_p2_matrix=predicted,
            )
            for policy_name in POLICIES:
                plan = resolve_policy(policy_name=policy_name, bucket_rows=0).build_logical_plan(problem)
                audit = replay_and_audit_logical_plan(problem, plan)
                makespan = float(audit.get("makespan", plan.diagnostics.get("makespan", 0.0)))
                if variant == "zero_hint":
                    zero_baseline[policy_name] = makespan
                rows.append(
                    {
                        "layer_id": layer_id,
                        "policy_name": policy_name,
                        "p2_variant": variant,
                        "mean_makespan": makespan,
                        "relative_to_zero_hint": None,
                        "fallback_to_B_ratio": 1.0 if bool(plan.diagnostics.get("fallback_to_paired_b", False)) else 0.0,
                        "selected_U_ratio": 0.0 if bool(plan.diagnostics.get("fallback_to_paired_b", False)) else 1.0,
                        "forecast_remote_bytes": int(matrix_remote_bytes(predicted)),
                        "p2_sensitivity_score": 0.0,
                        "selected_policy_distribution": {str(plan.diagnostics.get("selected_policy", plan.policy_name)): 1},
                    }
                )
        for row in rows:
            if row["layer_id"] != layer_id:
                continue
            zero = zero_baseline.get(str(row["policy_name"]))
            if zero not in (None, 0.0):
                row["relative_to_zero_hint"] = float((float(row["mean_makespan"]) - float(zero)) / float(zero))
                row["p2_sensitivity_score"] = float(-row["relative_to_zero_hint"])
    summary: list[dict[str, Any]] = []
    for policy_name in POLICIES:
        policy_rows = [row for row in rows if row["policy_name"] == policy_name]
        by_variant = {}
        for variant in variants:
            current = [row for row in policy_rows if row["p2_variant"] == variant]
            if not current:
                continue
            rel_values = [float(row["relative_to_zero_hint"]) for row in current if row["relative_to_zero_hint"] is not None]
            by_variant[variant] = {
                "policy_name": policy_name,
                "p2_variant": variant,
                "mean_makespan": statistics.mean([float(row["mean_makespan"]) for row in current]),
                "relative_to_zero_hint": statistics.mean(rel_values) if rel_values else None,
                "fallback_to_B_ratio": statistics.mean([float(row["fallback_to_B_ratio"]) for row in current]),
                "selected_U_ratio": statistics.mean([float(row["selected_U_ratio"]) for row in current]),
                "forecast_remote_bytes": statistics.mean([float(row["forecast_remote_bytes"]) for row in current]),
                "p2_sensitivity_score": statistics.mean([float(row["p2_sensitivity_score"]) for row in current]),
                "selected_policy_distribution": current[0]["selected_policy_distribution"],
            }
        summary.extend(by_variant.values())
    actual_rows = [row for row in summary if row["p2_variant"] == "actual_trace"]
    perfect_rows = [row for row in summary if row["p2_variant"] == "perfect_trace"]
    amplified_rows = [row for row in summary if row["p2_variant"] == "amplified_actual_4x"]
    shuffled_rows = [row for row in summary if row["p2_variant"] == "shuffled_actual"]
    eps = 1e-6
    actual_improves = any((row.get("relative_to_zero_hint") or 0.0) < -eps for row in actual_rows)
    perfect_improves = any((row.get("relative_to_zero_hint") or 0.0) < -eps for row in perfect_rows)
    amplified_improves = any((row.get("relative_to_zero_hint") or 0.0) < -eps for row in amplified_rows)
    shuffled_degrades = any((row.get("relative_to_zero_hint") or 0.0) > eps for row in shuffled_rows)
    actual_near_zero = all(abs(float(row.get("relative_to_zero_hint") or 0.0)) <= eps for row in actual_rows) if actual_rows else True
    amplified_near_zero = all(abs(float(row.get("relative_to_zero_hint") or 0.0)) <= eps for row in amplified_rows) if amplified_rows else True
    likely_reason = "prediction_can_help_if_accurate"
    if actual_near_zero and amplified_near_zero:
        likely_reason = "p2_signal_unused_or_too_weak"
    elif amplified_improves and not actual_improves:
        likely_reason = "real_p2_signal_weak"
    elif shuffled_degrades and not actual_improves:
        likely_reason = "safe_fallback_masks_bad_prediction"
    return {
        "fixture_dir": str(fixture_dir),
        "rows": rows,
        "summary": summary,
        "p2_signal_used": bool(actual_improves or perfect_improves or amplified_improves),
        "p2_signal_strength": None
        if not amplified_rows
        else statistics.mean([abs(float(row["relative_to_zero_hint"] or 0.0)) for row in amplified_rows]),
        "likely_reason_prediction_no_gain": likely_reason,
        "shuffled_degradation_detected": bool(shuffled_degrades),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P2 Sensitivity Replay",
        "",
        f"- fixture_dir: `{payload['fixture_dir']}`",
        f"- p2_signal_used: `{payload['p2_signal_used']}`",
        f"- likely_reason_prediction_no_gain: `{payload['likely_reason_prediction_no_gain']}`",
        "",
        "| Policy | P2 variant | Mean makespan | Rel to zero | fallback/B | selected U | forecast bytes | sensitivity |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        rel = "-" if row["relative_to_zero_hint"] is None else f"{100.0 * float(row['relative_to_zero_hint']):.2f}%"
        lines.append(
            f"| {row['policy_name']} | {row['p2_variant']} | {float(row['mean_makespan']):.0f} | {rel} | "
            f"{100.0 * float(row['fallback_to_B_ratio']):.2f}% | {100.0 * float(row['selected_U_ratio']):.2f}% | "
            f"{float(row['forecast_remote_bytes']):.0f} | {float(row['p2_sensitivity_score']):.4f} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["POLICIES", "render_markdown", "run_p2_sensitivity_replay"]
