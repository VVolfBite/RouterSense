"""Evaluate model-agnostic P2 forecast consumption under controlled corruption.

This experiment does not claim to reproduce FATE.  It perturbs true rank-level
P2 matrices while preserving each source row's assignment total, then measures
how much of the true-hint scheduling opportunity is captured by a scheduler.
The purpose is to evaluate the *forecast-to-scheduler interface* before full
expert-logit traces are available.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

from rs.runtime.offline.p2_information_value import simulate_p2_information


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def corrupt_row_preserving(
    matrix: list[list[int]],
    *,
    corruption_rate: float,
    seed: int,
) -> list[list[int]]:
    """Move a controlled fraction of assignments to random destinations.

    Every source-row total is preserved exactly.  The perturbation is expressed
    only in rank-level traffic geometry and therefore has no model-specific
    assumptions about expert count, layer identity, or routing architecture.
    """

    rate = max(0.0, min(1.0, float(corruption_rate)))
    rng = random.Random(int(seed))
    size = len(matrix)
    output = [[0 for _ in range(size)] for _ in range(size)]
    for source, row in enumerate(matrix):
        moved = 0
        for destination, raw_count in enumerate(row):
            count = int(raw_count)
            moved_from_cell = sum(1 for _ in range(count) if rng.random() < rate)
            output[source][destination] += count - moved_from_cell
            moved += moved_from_cell
        for _ in range(moved):
            output[source][rng.randrange(size)] += 1
        assert sum(output[source]) == sum(int(value) for value in row)
    return output


def _remote_l1_relative(truth: list[list[int]], forecast: list[list[int]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for source, truth_row in enumerate(truth):
        for destination, truth_value in enumerate(truth_row):
            if source == destination:
                continue
            numerator += abs(float(truth_value) - float(forecast[source][destination]))
            denominator += float(truth_value)
    return numerator / max(denominator, 1.0)


def _select_stratified(
    instances: list[dict[str, Any]],
    *,
    per_vep: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for virtual_ep_size in sorted({int(row["virtual_ep_size"]) for row in instances}):
        rows = sorted(
            (
                row
                for row in instances
                if int(row["virtual_ep_size"]) == virtual_ep_size
                and bool(row.get("p2_available", False))
            ),
            key=lambda row: (
                str(row["sample_id"]),
                int(row["layer_id"]),
                str(row["traffic_instance_id"]),
            ),
        )
        if len(rows) <= per_vep:
            selected.extend(rows)
            continue
        # Even spacing avoids selecting only the first prompts/layers.
        indexes = {
            round(index * (len(rows) - 1) / max(per_vep - 1, 1))
            for index in range(per_vep)
        }
        selected.extend(rows[index] for index in sorted(indexes))
    return selected


def _evaluate_one(task: tuple[dict[str, Any], str, tuple[float, ...], int]) -> list[dict[str, Any]]:
    item, family_id, corruption_rates, replicate_count = task
    p0 = item["P0_dispatch_matrix"]
    p1 = item["P1_return_matrix"]
    truth = item["P2_next_layer_dispatch_matrix"]
    common = {
        "p0_dispatch_matrix": p0,
        "p1_return_matrix": p1,
        "p2_truth_matrix": truth,
        "family_id": family_id,
    }
    reactive = simulate_p2_information(**common, information_mode="reactive")
    true_hint = simulate_p2_information(
        **common,
        information_mode="predicted",
        p2_forecast_matrix=truth,
        prediction_confidence=1.0,
    )
    perfect_window = simulate_p2_information(**common, information_mode="perfect")
    if not (reactive.valid and true_hint.valid and perfect_window.valid):
        raise RuntimeError(
            f"invalid base replay for {item['traffic_instance_id']} / {family_id}"
        )
    rows: list[dict[str, Any]] = []
    for corruption_rate in corruption_rates:
        for replicate in range(replicate_count):
            seed = _stable_seed(
                item["traffic_instance_id"],
                family_id,
                corruption_rate,
                replicate,
            )
            forecast = corrupt_row_preserving(
                truth,
                corruption_rate=corruption_rate,
                seed=seed,
            )
            predicted = simulate_p2_information(
                **common,
                information_mode="predicted",
                p2_forecast_matrix=forecast,
                prediction_confidence=max(0.0, 1.0 - float(corruption_rate)),
            )
            if not predicted.valid:
                raise RuntimeError(
                    f"invalid predicted replay for {item['traffic_instance_id']} / "
                    f"{family_id} / {corruption_rate}"
                )
            reactive_makespan = float(reactive.makespan)
            true_hint_makespan = float(true_hint.makespan)
            predicted_makespan = float(predicted.makespan)
            perfect_gain = reactive_makespan - true_hint_makespan
            predicted_gain = reactive_makespan - predicted_makespan
            rows.append(
                {
                    "traffic_instance_id": item["traffic_instance_id"],
                    "sample_id": item["sample_id"],
                    "layer_id": int(item["layer_id"]),
                    "virtual_ep_size": int(item["virtual_ep_size"]),
                    "family_id": family_id,
                    "corruption_rate": float(corruption_rate),
                    "replicate": int(replicate),
                    "seed": int(seed),
                    "remote_l1_relative": _remote_l1_relative(truth, forecast),
                    "reactive_makespan": reactive_makespan,
                    "true_hint_makespan": true_hint_makespan,
                    "perfect_window_makespan": float(perfect_window.makespan),
                    "predicted_makespan": predicted_makespan,
                    "true_hint_improvement_vs_reactive_pct": (
                        100.0 * perfect_gain / reactive_makespan
                        if reactive_makespan > 0.0
                        else 0.0
                    ),
                    "predicted_improvement_vs_reactive_pct": (
                        100.0 * predicted_gain / reactive_makespan
                        if reactive_makespan > 0.0
                        else 0.0
                    ),
                    "prediction_regret_vs_true_hint_pct": (
                        100.0 * (predicted_makespan - true_hint_makespan) / true_hint_makespan
                        if true_hint_makespan > 0.0
                        else 0.0
                    ),
                    "capture_ratio": (
                        predicted_gain / perfect_gain if perfect_gain > 1e-9 else None
                    ),
                    "predicted_runtime_ms": float(predicted.planning_time_ms),
                }
            )
    return rows


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = [float(value) for value in values]
    return {
        "n": len(rows),
        "mean": statistics.mean(rows) if rows else None,
        "median": statistics.median(rows) if rows else None,
        "p10": _percentile(rows, 0.10),
        "p90": _percentile(rows, 0.90),
        "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups = sorted(
        {
            (str(row["family_id"]), float(row["corruption_rate"]))
            for row in records
        }
    )
    for family_id, corruption_rate in groups:
        rows = [
            row
            for row in records
            if row["family_id"] == family_id
            and float(row["corruption_rate"]) == corruption_rate
        ]
        captures = [
            float(row["capture_ratio"])
            for row in rows
            if row["capture_ratio"] is not None
        ]
        improvements = [float(row["predicted_improvement_vs_reactive_pct"]) for row in rows]
        output[f"{family_id}@{corruption_rate:.3f}"] = {
            "family_id": family_id,
            "corruption_rate": corruption_rate,
            "remote_l1_relative": _stats(row["remote_l1_relative"] for row in rows),
            "true_hint_improvement_vs_reactive_pct": _stats(
                row["true_hint_improvement_vs_reactive_pct"] for row in rows
            ),
            "predicted_improvement_vs_reactive_pct": _stats(improvements),
            "prediction_regret_vs_true_hint_pct": _stats(
                row["prediction_regret_vs_true_hint_pct"] for row in rows
            ),
            "capture_ratio": _stats(captures),
            "non_worse_than_reactive_pct": (
                100.0 * sum(value >= -1e-9 for value in improvements) / len(improvements)
            ),
            "regression_vs_reactive_pct": (
                100.0 * sum(value < -1e-9 for value in improvements) / len(improvements)
            ),
            "planning_runtime_ms": _stats(row["predicted_runtime_ms"] for row in rows),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=("rsbc", "rscf"))
    parser.add_argument("--corruption-rates", nargs="+", type=float, default=(0.10, 0.25, 0.40))
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--per-vep", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    instances = json.loads(args.instances.read_text(encoding="utf-8"))
    selected = _select_stratified(instances, per_vep=max(1, int(args.per_vep)))
    tasks = [
        (item, str(family_id), tuple(args.corruption_rates), max(1, int(args.replicates)))
        for item in selected
        for family_id in args.families
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        nested = list(executor.map(_evaluate_one, tasks, chunksize=1))
    records = [row for group in nested for row in group]
    artifact = {
        "schema_version": "p2_forecast_heuristic_evaluation.v1",
        "input_instance_count": len(instances),
        "selected_instance_count": len(selected),
        "selection_per_vep": int(args.per_vep),
        "families": list(args.families),
        "corruption_rates": list(args.corruption_rates),
        "replicates": int(args.replicates),
        "forecast_model": (
            "row-total-preserving rank-destination corruption; development POC only; "
            "not a FATE accuracy claim"
        ),
        "summary": _summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))


if __name__ == "__main__":
    main()
