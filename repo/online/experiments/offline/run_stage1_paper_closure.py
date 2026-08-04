#!/usr/bin/env python3
"""Unified offline stage1 closure runner for baselines, oracle, predictors, and schedule regret."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from experiments.offline.replay_fixture_policy_study import _build_problem
from rs.runtime.online.megatron_ep.async_release.runtime_projection import host_project_safe_selection
from rs.scheduling import resolve_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for stage1 paper closure")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src:." if not existing else f"src:.:{existing}"
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=False)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _pct(base: float | None, value: float | None) -> float | None:
    if base in (None, 0.0) or value is None:
        return None
    return float((float(base) - float(value)) / float(base))


def _map_predictor_name(name: str) -> str:
    return {
        "fate_style_history": "history_ema",
        "fate_style_linear": "history_linear_trend",
        "perfect_trace_oracle": "oracle_traffic",
        "actual_trace_oracle": "oracle_traffic",
        "perfect_trace": "oracle_traffic",
        "actual_trace": "oracle_traffic",
    }.get(name, name)


def _contiguous_split(layer_ids: list[str], *, train_ratio: float, validation_ratio: float) -> dict[str, set[str]]:
    ordered = sorted(layer_ids, key=lambda item: int(item))
    n = len(ordered)
    train_n = max(1, int(math.floor(n * train_ratio))) if n else 0
    valid_n = max(1, int(math.floor(n * validation_ratio))) if n - train_n > 1 else max(0, n - train_n - 1)
    train = set(ordered[:train_n])
    valid = set(ordered[train_n : train_n + valid_n])
    test = set(ordered[train_n + valid_n :])
    if not test and ordered:
        last = ordered[-1]
        valid.discard(last)
        test.add(last)
    return {"train": train, "validation": valid, "test": test}


def _predictor_overhead_rank(name: str) -> int:
    order = {
        "zero_hint": 0,
        "copy_current_dispatch": 1,
        "history_ema": 2,
        "history_linear_trend": 3,
        "oracle_traffic": 99,
    }
    return order.get(name, 50)


def _summarize_predictors(
    rows: list[dict[str, Any]],
    split: dict[str, set[str]],
    *,
    allowed_predictors: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_rows = [
        {
            **row,
            "predictor_track_name": _map_predictor_name(str(row["predictor_name"])),
            "split": next((name for name, ids in split.items() if str(row["source_layer_id"]) in ids), "test"),
        }
        for row in rows
        if bool(row.get("evaluation_eligible"))
    ]
    online_rows = [row for row in normalized_rows if str(row["predictor_track_name"]) in allowed_predictors]
    per_predictor_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    per_layer_policy_best: dict[tuple[str, str, str], float] = {}
    for row in online_rows:
        key = (str(row["predictor_track_name"]), str(row["split"]))
        per_predictor_split.setdefault(key, []).append(row)
        layer_policy_key = (str(row["split"]), str(row["policy_name"]), str(row["source_layer_id"]))
        current = per_layer_policy_best.get(layer_policy_key)
        value = float(row["safe_makespan"])
        if current is None or value < current:
            per_layer_policy_best[layer_policy_key] = value

    table_rows: list[dict[str, Any]] = []
    for (predictor_name, split_name), items in sorted(per_predictor_split.items()):
        regret_values = []
        for row in items:
            best = per_layer_policy_best[(split_name, str(row["policy_name"]), str(row["source_layer_id"]))]
            regret_values.append((float(row["safe_makespan"]) - float(best)) / max(float(best), 1.0))
        table_rows.append(
            {
                "predictor_name": predictor_name,
                "split": split_name,
                "row_count": len(items),
                "mean_relative_l1": _mean([float(row["prediction_relative_l1_error"]) for row in items]),
                "mean_cosine": _mean([float(row["prediction_cosine_similarity"]) for row in items]),
                "mean_row_sum_error": _mean([float(row.get("row_sum_error", 0.0)) for row in items]),
                "mean_col_sum_error": _mean([float(row.get("col_sum_error", 0.0)) for row in items]),
                "max_port_load_error_proxy": max(
                    [
                        max(float(row.get("row_sum_error", 0.0)), float(row.get("col_sum_error", 0.0)))
                        for row in items
                    ],
                    default=0.0,
                ),
                "heavy_edge_recall": _mean([float(row.get("topk_edge_overlap", 0.0)) for row in items]),
                "bottleneck_source_recall_proxy": _mean([float(row.get("topk_edge_overlap", 0.0)) for row in items]),
                "bottleneck_destination_recall_proxy": _mean([float(row.get("nonzero_recall", 0.0)) for row in items]),
                "schedule_regret": _mean(regret_values),
                "safe_fallback_rate": _mean([1.0 if bool(row.get("fallback_to_B")) else 0.0 for row in items]),
                "prediction_overhead_rank": _predictor_overhead_rank(predictor_name),
                "online_eligible": True,
            }
        )
    validation_rows = [row for row in table_rows if row["split"] == "validation"]
    if not validation_rows:
        raise RuntimeError("missing validation rows for predictor selection")
    selected = min(
        validation_rows,
        key=lambda item: (
            float(item["schedule_regret"] if item["schedule_regret"] is not None else 1e9),
            int(item["prediction_overhead_rank"]),
            float(item["mean_relative_l1"] if item["mean_relative_l1"] is not None else 1e9),
        ),
    )
    selected_name = str(selected["predictor_name"])
    test_rows = [row for row in table_rows if row["split"] == "test"]
    selected_test = next((row for row in test_rows if str(row["predictor_name"]) == selected_name), None)
    selection = {
        "selected_predictor": selected_name,
        "selection_reason": "min_validation_schedule_regret_then_overhead_then_matrix_error",
        "validation_metrics": selected,
        "held_out_test_metrics": selected_test,
    }

    taxonomy_rows: list[dict[str, Any]] = []
    by_layer_policy_predictor = {(str(r["source_layer_id"]), str(r["policy_name"]), str(r["predictor_track_name"])): r for r in online_rows}
    layer_policy_keys = sorted({(str(r["source_layer_id"]), str(r["policy_name"])) for r in online_rows}, key=lambda item: (int(item[0]), item[1]))
    for layer_id, policy_name in layer_policy_keys:
        zero = by_layer_policy_predictor.get((layer_id, policy_name, "zero_hint"))
        pred = by_layer_policy_predictor.get((layer_id, policy_name, selected_name))
        oracle = by_layer_policy_predictor.get((layer_id, policy_name, "oracle_traffic"))
        if zero is None or pred is None:
            continue
        zero_ms = float(zero["safe_makespan"])
        pred_ms = float(pred["safe_makespan"])
        oracle_ms = None if oracle is None else float(oracle["safe_makespan"])
        label = "PREDICTION_NEUTRAL"
        eps = 1e-6 * max(zero_ms, 1.0)
        if oracle_ms is not None and oracle_ms < zero_ms - eps and pred_ms < zero_ms - eps:
            label = "PREDICTION_HELPED"
        elif oracle_ms is not None and oracle_ms >= zero_ms - eps and oracle_ms <= zero_ms + eps:
            label = "ORACLE_ALSO_NO_GAIN"
        elif oracle_ms is not None and oracle_ms < zero_ms - eps and pred_ms > zero_ms + eps:
            label = "PREDICTION_HURT"
        elif oracle_ms is not None and oracle_ms < zero_ms - eps and bool(pred.get("fallback_to_B")):
            label = "SAFE_FALLBACK_MASKED_GAIN"
        elif oracle_ms is not None and oracle_ms > zero_ms + eps:
            label = "ORACLE_GAIN_MODEL_FAILS"
        taxonomy_rows.append(
            {
                "layer_id": layer_id,
                "policy_name": policy_name,
                "selected_predictor": selected_name,
                "zero_hint_makespan": zero_ms,
                "predicted_makespan": pred_ms,
                "oracle_makespan": oracle_ms,
                "prediction_schedule_regret": None if zero_ms == 0 else float((pred_ms - zero_ms) / zero_ms),
                "classification": label,
            }
        )
    return (
        [row for row in table_rows if row["split"] == "validation"],
        [row for row in table_rows if row["split"] == "test"],
        selection,
        taxonomy_rows,
        normalized_rows,
    )


def _host_projection_rows(fixture_dir: Path) -> list[dict[str, Any]]:
    fixtures = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda path: int(path.stem.split("_")[-1]))
    families = [
        ("gated_greedy", "B_gated_greedy_maximal", "U_gated_greedy_maximal", "RS_safe_gated_greedy"),
        ("barrier_criticality_matching", "B_barrier_criticality_matching", "U_barrier_criticality_global_matching", "RS_safe_barrier_criticality"),
    ]
    rows: list[dict[str, Any]] = []
    for family, b_name, u_name, safe_name in families:
        ideal_raw = []
        host_raw = []
        ideal_b = []
        host_b = []
        for fixture_path in fixtures:
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            problem = _build_problem(fixture, mode="runtime_lookahead", p2_source="zero_hint", expert_compute_delay=0.0)
            raw_u = resolve_policy(policy_name=u_name, bucket_rows=0).build_logical_plan(problem)
            paired_b = resolve_policy(policy_name=b_name, bucket_rows=0).build_logical_plan(problem)
            projected = host_project_safe_selection(raw_u_plan=raw_u, paired_b_plan=paired_b)
            ideal_raw.append(float(projected["ideal_raw_u_estimated_makespan"]))
            host_raw.append(float(projected["host_projected_raw_u_estimated_makespan"]))
            ideal_b.append(float(projected["ideal_paired_b_estimated_makespan"]))
            host_b.append(float(projected["host_projected_paired_b_estimated_makespan"]))
        rows.append(
            {
                "heuristic_family": family,
                "raw_u_policy": u_name,
                "paired_b_policy": b_name,
                "safe_policy": safe_name,
                "ideal_raw_u_makespan": _mean(ideal_raw),
                "host_projected_raw_u_makespan": _mean(host_raw),
                "ideal_paired_b_makespan": _mean(ideal_b),
                "host_projected_paired_b_makespan": _mean(host_b),
                "ideal_u_vs_b": _pct(_mean(ideal_b), _mean(ideal_raw)),
                "host_projected_u_vs_b": _pct(_mean(host_b), _mean(host_raw)),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    config = _load_yaml(Path(args.config))
    fixture_dir = ROOT / str(config["fixture_dir"])
    output_dir = ROOT / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        (
            output_dir / "replay_suite_summary.json",
            [
                sys.executable,
                "experiments/offline/run_replay_fixture_policy_suite.py",
                "--fixture-dir",
                str(fixture_dir),
                "--output-summary",
                str(output_dir / "replay_suite_summary.json"),
                "--output-summary-md",
                str(output_dir / "replay_suite_summary.md"),
            ],
        ),
        (
            output_dir / "oracle_gap_raw.json",
            [
                sys.executable,
                "experiments/offline/run_oracle_gap_replay.py",
                "--fixture-dir",
                str(fixture_dir),
                "--output-summary",
                str(output_dir / "oracle_gap_raw.json"),
            ],
        ),
        (
            output_dir / "prediction_replay_raw.json",
            [
                sys.executable,
                "experiments/offline/run_prediction_replay_suite.py",
                "--fixture-dir",
                str(fixture_dir),
                "--traffic-calibration",
                str(config.get("traffic_calibration", "none")),
                "--output-summary",
                str(output_dir / "prediction_replay_raw.json"),
                "--output-summary-md",
                str(output_dir / "prediction_replay_raw.md"),
            ],
        ),
        (
            output_dir / "p2_sensitivity_raw.json",
            [
                sys.executable,
                "experiments/offline/run_p2_sensitivity_replay.py",
                "--fixture-dir",
                str(fixture_dir),
                "--output-summary",
                str(output_dir / "p2_sensitivity_raw.json"),
                "--output-summary-md",
                str(output_dir / "p2_sensitivity_raw.md"),
            ],
        ),
        (
            output_dir / "p2_bridge_diagnosis" / "common_core_b_u_replay.json",
            [
                sys.executable,
                "experiments/offline/run_p2_bridge_async_ar0_diagnosis.py",
                "--fixture-dir",
                str(fixture_dir),
                "--output-dir",
                str(output_dir / "p2_bridge_diagnosis"),
            ],
        ),
    ]
    command_rows = []
    for expected_output, cmd in commands:
        name = Path(cmd[1]).stem
        if expected_output.exists() and expected_output.stat().st_size > 0:
            command_rows.append({"command": cmd, "returncode": 0, "cached": True})
            continue
        proc = _run(cmd, cwd=ROOT)
        (output_dir / f"{name}.stdout.log").write_text(proc.stdout, encoding="utf-8")
        (output_dir / f"{name}.stderr.log").write_text(proc.stderr, encoding="utf-8")
        command_rows.append({"command": cmd, "returncode": int(proc.returncode), "cached": False})
        if proc.returncode != 0:
            raise SystemExit(f"{name} failed with return code {proc.returncode}")

    replay = json.loads((output_dir / "replay_suite_summary.json").read_text(encoding="utf-8"))
    oracle_gap = json.loads((output_dir / "oracle_gap_raw.json").read_text(encoding="utf-8"))
    prediction = json.loads((output_dir / "prediction_replay_raw.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((output_dir / "p2_sensitivity_raw.json").read_text(encoding="utf-8"))
    common_core = json.loads((output_dir / "p2_bridge_diagnosis" / "common_core_b_u_replay.json").read_text(encoding="utf-8"))

    baseline_rows: list[dict[str, Any]] = []
    for row in replay["table_a"]["summary"]:
        baseline_rows.append(
            {
                "comparison_family": "phase_sync_compatible",
                "strict_common_core": False,
                "backend_name": "runtime_lookahead",
                "only_scope_differs": False,
                **row,
            }
        )
    for row in replay["table_b"]["summary"]:
        baseline_rows.append(
            {
                "comparison_family": "execution_window_joint",
                "strict_common_core": False,
                "backend_name": "execution_window",
                "only_scope_differs": False,
                **row,
            }
        )
    for row in replay["paired_b_vs_u"]["summary"]:
        baseline_rows.append(
            {
                "comparison_family": "legacy_paired_family",
                "strict_common_core": False,
                "backend_name": "paired_family",
                "only_scope_differs": False,
                **row,
            }
        )

    _write_csv(output_dir / "baseline_summary.csv", baseline_rows)
    _write_json(output_dir / "baseline_summary.json", {"rows": baseline_rows})

    per_layer_rows = list(replay["table_a"]["rows"]) + list(replay["table_b"]["rows"]) + list(prediction["rows"])
    _write_csv(output_dir / "per_layer_results.csv", per_layer_rows)

    shared_core_rows = []
    for row in common_core["rows"]:
        shared_core_rows.append(
            {
                "comparison_family": "shared_core_b_u",
                "strict_common_core": False,
                "backend_name": row["matching_backend"],
                "only_scope_differs": bool(row["joint_only_difference"]),
                **row,
            }
        )
    _write_csv(output_dir / "shared_core_b_u.csv", shared_core_rows)
    _write_md(
        output_dir / "shared_core_b_u.md",
        "# Shared-Core B/U\n\n" + "\n".join(
            f"- `{row['heuristic_family']}`: U vs B `{100.0 * float(row['relative_improvement_u_vs_b']):.2f}%`, safe vs B `{100.0 * float(row['relative_improvement_safe_vs_b']):.2f}%`"
            for row in shared_core_rows
        )
        + "\n",
    )

    _write_csv(output_dir / "legacy_paired_family.csv", list(replay["paired_b_vs_u"]["summary"]))

    oracle_rows = []
    for row in oracle_gap["small_fixture_rows"]:
        oracle_rows.append({"oracle_scope": "small_fixture", **row})
    oracle_rows.append({"oracle_scope": "gap_summary", **oracle_gap["oracle_gap_small_fixture_summary"]})
    _write_csv(output_dir / "oracle_gap.csv", oracle_rows)
    _write_md(
        output_dir / "oracle_gap.md",
        "# Oracle Gap\n\n"
        f"- O_local_definition: `{oracle_gap['O_local_definition']}`\n"
        f"- O_joint_definition: `{oracle_gap['O_joint_definition']}`\n"
        f"- O_joint_vs_O_local_gap: `{oracle_gap['oracle_gap_small_fixture_summary']['O_joint_vs_O_local_gap']}`\n"
        f"- B_gap_to_O_local: `{oracle_gap['oracle_gap_small_fixture_summary']['B_gap_to_O_local']}`\n",
    )

    split = _contiguous_split(
        sorted({str(row["source_layer_id"]) for row in prediction["rows"]}, key=lambda item: int(item)),
        train_ratio=float(config["predictor_split"]["train_ratio"]),
        validation_ratio=float(config["predictor_split"]["validation_ratio"]),
    )
    allowed_predictors = {_map_predictor_name(str(name)) for name in list(config.get("predictor_candidates", []))}
    validation_rows, test_rows, selection, taxonomy_rows, normalized_prediction_rows = _summarize_predictors(
        list(prediction["rows"]),
        split,
        allowed_predictors=allowed_predictors,
    )
    _write_csv(output_dir / "predictor_validation.csv", validation_rows)
    _write_csv(output_dir / "predictor_test.csv", test_rows)
    _write_json(output_dir / "predictor_selection.json", selection)
    _write_csv(output_dir / "prediction_schedule_regret.csv", taxonomy_rows)

    taxonomy_counts: dict[str, int] = {}
    for row in taxonomy_rows:
        taxonomy_counts[str(row["classification"])] = taxonomy_counts.get(str(row["classification"]), 0) + 1
    taxonomy_payload = {"rows": taxonomy_rows, "counts": taxonomy_counts}
    _write_csv(output_dir / "prediction_failure_taxonomy.csv", taxonomy_rows)

    host_projection_rows = _host_projection_rows(fixture_dir)
    _write_csv(output_dir / "host_projection_gap.csv", host_projection_rows)

    baseline_best = min(
        (row for row in baseline_rows if row.get("mean_makespan") is not None and row.get("comparison_family") == "phase_sync_compatible"),
        key=lambda item: float(item["mean_makespan"]),
    )
    shared_core_best = min(shared_core_rows, key=lambda item: float(item["mean_makespan_safe"]))
    predictor_selected_test = selection.get("held_out_test_metrics") or {}
    oracle_actual = [
        row
        for row in normalized_prediction_rows
        if str(row["predictor_track_name"]) == "oracle_traffic" and str(row["split"]) == "test"
    ]
    selected_predictor = str(selection["selected_predictor"])
    selected_predictor_rows = [row for row in normalized_prediction_rows if str(row["predictor_track_name"]) == selected_predictor and str(row["split"]) == "test"]
    zero_rows = [row for row in normalized_prediction_rows if str(row["predictor_track_name"]) == "zero_hint" and str(row["split"]) == "test"]
    zero_mean = _mean([float(row["safe_makespan"]) for row in zero_rows])
    selected_mean = _mean([float(row["safe_makespan"]) for row in selected_predictor_rows])
    oracle_mean = _mean([float(row["safe_makespan"]) for row in oracle_actual if str(row["split"]) == "test"])

    final_summary = {
        "fixture_dir": str(fixture_dir),
        "output_dir": str(output_dir),
        "commands": command_rows,
        "strongest_baseline": baseline_best["policy_name"],
        "strict_common_core_completed": False,
        "joint_only_gain": [
            {
                "heuristic_family": row["heuristic_family"],
                "u_vs_b_makespan_reduction_percent": None if row["relative_improvement_u_vs_b"] is None else float(-row["relative_improvement_u_vs_b"]) * 100.0,
                "safe_vs_b_makespan_reduction_percent": None if row["relative_improvement_safe_vs_b"] is None else float(-row["relative_improvement_safe_vs_b"]) * 100.0,
            }
            for row in shared_core_rows
        ],
        "safe_u_relative_to_strongest_b": {
            "main_safe_u_policy": replay.get("main_safe_u_policy"),
            "main_safe_u_improvement_pct": replay.get("main_safe_u_improvement_pct"),
        },
        "oracle_gap": oracle_gap["oracle_gap_small_fixture_summary"],
        "oracle_gap_reduction_percent": {
            "O_joint_vs_O_local": None if oracle_gap["oracle_gap_small_fixture_summary"]["O_joint_vs_O_local_gap"] is None else float(-oracle_gap["oracle_gap_small_fixture_summary"]["O_joint_vs_O_local_gap"]) * 100.0,
            "B_gap_to_O_local": None if oracle_gap["oracle_gap_small_fixture_summary"]["B_gap_to_O_local"] is None else float(oracle_gap["oracle_gap_small_fixture_summary"]["B_gap_to_O_local"]) * 100.0,
            "raw_U_gap_to_O_joint": None if oracle_gap["oracle_gap_small_fixture_summary"]["raw_U_gap_to_O_joint"] is None else float(oracle_gap["oracle_gap_small_fixture_summary"]["raw_U_gap_to_O_joint"]) * 100.0,
            "safe_U_gap_to_O_joint": None if oracle_gap["oracle_gap_small_fixture_summary"]["safe_U_gap_to_O_joint"] is None else float(oracle_gap["oracle_gap_small_fixture_summary"]["safe_U_gap_to_O_joint"]) * 100.0,
        },
        "selected_predictor": selection["selected_predictor"],
        "selected_predictor_reason": selection["selection_reason"],
        "held_out_prediction_error": predictor_selected_test.get("mean_relative_l1"),
        "held_out_schedule_regret": predictor_selected_test.get("schedule_regret"),
        "oracle_prediction_schedule_gain_vs_zero": None if zero_mean in (None, 0.0) or oracle_mean is None else float((zero_mean - oracle_mean) / zero_mean),
        "selected_prediction_schedule_gain_vs_zero": None if zero_mean in (None, 0.0) or selected_mean is None else float((zero_mean - selected_mean) / zero_mean),
        "prediction_failure_taxonomy": taxonomy_counts,
        "p2_signal_used": sensitivity["p2_signal_used"],
        "p2_signal_strength": sensitivity["p2_signal_strength"],
        "likely_reason_prediction_no_gain": sensitivity["likely_reason_prediction_no_gain"],
        "host_projection_gap": host_projection_rows,
        "exact_oracle_sample_count": len(oracle_gap.get("small_fixture_rows", [])),
    }
    _write_json(output_dir / "final_offline_summary.json", final_summary)

    fixture_paths = sorted(fixture_dir.glob("*.json"))
    fixture_manifest = {
        "fixture_dir": str(fixture_dir),
        "fixture_file_count": len(fixture_paths),
        "fixture_files": [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in fixture_paths
        ],
        "replay_command": f"python experiments/offline/run_stage1_paper_closure.py --config {args.config}",
        "config_path": str(Path(args.config)),
    }
    _write_json(output_dir / "fixture_manifest.json", fixture_manifest)

    baseline_lines = [
        "# Stage1 Paper Closure Tables",
        "",
        "## Baselines",
        f"- strongest baseline: `{baseline_best['policy_name']}`",
        f"- execution-window best U: `{replay.get('execution_window_best_u')}`",
        "",
        "## Shared-Core B/U",
    ]
    baseline_lines.extend(
        f"- `{row['heuristic_family']}` joint-only gain (U vs B): `{(-100.0) * float(row['relative_improvement_u_vs_b']):.2f}%`"
        for row in shared_core_rows
    )
    baseline_lines.extend(
        [
            "",
            "## Oracle Gap",
            f"- O_joint_vs_O_local_gap: `{oracle_gap['oracle_gap_small_fixture_summary']['O_joint_vs_O_local_gap']}`",
            f"- O_joint_vs_O_local_makespan_reduction_percent: `{None if oracle_gap['oracle_gap_small_fixture_summary']['O_joint_vs_O_local_gap'] is None else (-100.0 * float(oracle_gap['oracle_gap_small_fixture_summary']['O_joint_vs_O_local_gap']))}`",
            "",
            "## Predictor Selection",
            f"- selected predictor: `{selection['selected_predictor']}`",
            f"- held-out schedule regret: `{predictor_selected_test.get('schedule_regret')}`",
            f"- held-out relative L1: `{predictor_selected_test.get('mean_relative_l1')}`",
            "",
            "## Scheduling Regret",
            f"- selected predictor gain vs zero: `{final_summary['selected_prediction_schedule_gain_vs_zero']}`",
            f"- oracle gain vs zero: `{final_summary['oracle_prediction_schedule_gain_vs_zero']}`",
            f"- taxonomy: `{taxonomy_counts}`",
            "",
        ]
    )
    _write_md(output_dir / "paper_ready_tables.md", "\n".join(baseline_lines) + "\n")
    _write_md(
        output_dir / "final_offline_conclusions.md",
        "# Final Offline Conclusions\n\n"
        f"- strongest phase-sync-compatible baseline remains `{baseline_best['policy_name']}`.\n"
        f"- shared-core proxy gains remain positive for the two main families, but `strict_common_core=false`.\n"
        f"- selected online-eligible predictor is `{selection['selected_predictor']}` by validation schedule regret.\n"
        f"- held-out predictor schedule regret is `{predictor_selected_test.get('schedule_regret')}`.\n"
        f"- selected predictor gain vs zero is `{final_summary['selected_prediction_schedule_gain_vs_zero']}`; oracle gain vs zero is `{final_summary['oracle_prediction_schedule_gain_vs_zero']}`.\n"
        f"- prediction failure taxonomy: `{taxonomy_counts}`.\n",
    )
    _write_md(
        output_dir / "baseline_summary.md",
        "# Baseline Summary\n\n"
        + "\n".join(
            "- "
            + f"`{row.get('policy_name', row.get('safe_U_algorithm', row.get('heuristic_family', 'unknown')))}"
            + f"` (`{row['comparison_family']}`): mean_makespan=`{row.get('mean_makespan', row.get('safe_U_makespan'))}`"
            for row in baseline_rows[:24]
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
