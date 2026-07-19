"""Run the unified scheduling/P2/prediction closure on a traffic package."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from rs.runtime.offline.scheduling_prediction_closure import (
    evaluate_exact_closure,
    evaluate_family_closure,
    stats,
    summarize_family_records,
)
from rs.runtime.offline.traffic_dataset import TrafficInstanceRecord, load_traffic_instances


Matrix = tuple[tuple[int, ...], ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--world-sizes", nargs="+", type=int, default=(2, 4, 8, 16))
    parser.add_argument("--families", nargs="+", default=("greedy_control", "gmwd", "rsbc", "fast_stage"))
    parser.add_argument("--max-per-world", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--forecast-mode", choices=("none", "truth", "corrupt", "json"), default="corrupt")
    parser.add_argument("--forecast-json", type=Path)
    parser.add_argument("--corruption-rate", type=float, default=0.20)
    parser.add_argument("--prediction-confidence", type=float, default=0.80)
    parser.add_argument("--exact-instance-count", type=int, default=24)
    parser.add_argument("--exact-time-limit-ms", type=int, default=5000)
    parser.add_argument("--include-records", action="store_true")
    return parser.parse_args()


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _corrupt(matrix: Matrix, *, rate: float, seed: int) -> Matrix:
    rng = random.Random(seed)
    size = len(matrix)
    output = [list(row) for row in matrix]
    for source, row in enumerate(matrix):
        move = int(round(sum(int(value) for value in row) * max(0.0, min(1.0, float(rate)))))
        cells = [destination for destination, value in enumerate(output[source]) if int(value) > 0]
        moved = 0
        while cells and moved < move:
            destination = cells[rng.randrange(len(cells))]
            if output[source][destination] <= 0:
                cells.remove(destination)
                continue
            output[source][destination] -= 1
            output[source][rng.randrange(size)] += 1
            moved += 1
    return tuple(tuple(int(value) for value in row) for row in output)


def _load_forecast_map(path: Path | None) -> dict[str, Matrix]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    output: dict[str, Matrix] = {}
    if isinstance(records, dict):
        iterable = ({"traffic_instance_id": key, "matrix": value} for key, value in records.items())
    elif isinstance(records, list):
        iterable = records
    else:
        raise ValueError("unsupported forecast JSON schema")
    for row in iterable:
        envelope = row.get("envelope") if isinstance(row, dict) else None
        matrix = row.get(
            "forecast_matrix",
            row.get(
                "mean_rows",
                row.get(
                    "matrix",
                    row.get("P2_predicted_matrix", None if envelope is None else envelope.get("mean_rows")),
                ),
            ),
        )
        if matrix is None:
            raise ValueError(f"forecast record for {row.get('traffic_instance_id')} has no mean matrix")
        output[str(row["traffic_instance_id"])] = tuple(tuple(int(value) for value in values) for values in matrix)
    return output


def _select(records: list[TrafficInstanceRecord], *, max_per_world: int) -> list[TrafficInstanceRecord]:
    if max_per_world <= 0:
        return records
    output: list[TrafficInstanceRecord] = []
    for world_size in sorted({record.world_size for record in records}):
        rows = sorted(
            (record for record in records if record.world_size == world_size),
            key=lambda record: (record.model_id, record.sample_id, record.layer_id, record.traffic_instance_id),
        )
        count = min(max_per_world, len(rows))
        indexes = sorted({round(index * (len(rows) - 1) / max(count - 1, 1)) for index in range(count)})
        output.extend(rows[index] for index in indexes)
    return output


def _forecast_for(
    record: TrafficInstanceRecord,
    *,
    mode: str,
    forecast_map: dict[str, Matrix],
    corruption_rate: float,
) -> Matrix | None:
    if mode == "none":
        return None
    if mode == "truth":
        return record.p2
    if mode == "json":
        try:
            return forecast_map[record.traffic_instance_id]
        except KeyError as exc:
            raise ValueError(f"missing forecast for {record.traffic_instance_id}") from exc
    return _corrupt(
        record.p2,
        rate=corruption_rate,
        seed=_stable_seed(record.traffic_instance_id, corruption_rate),
    )


def _family_task(task: tuple[TrafficInstanceRecord, str, Matrix | None, float]):
    record, family_id, forecast, confidence = task
    return evaluate_family_closure(
        record,
        family_id=family_id,
        p2_forecast=forecast,
        prediction_confidence=confidence,
    )


def _exact_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["reactive"]["valid"] and row["perfect"]["valid"]]
    predicted_valid = [row for row in valid if row.get("predicted") and row["predicted"]["valid"]]
    return {
        "instance_count": len(rows),
        "valid_reactive_perfect_count": len(valid),
        "valid_predicted_count": len(predicted_valid),
        "perfect_p2_information_value_pct": stats(
            100.0 * row["metrics"]["perfect_p2_information_value"]
            for row in valid
            if row["metrics"]["perfect_p2_information_value"] is not None
        ),
        "predicted_p2_information_value_pct": stats(
            100.0 * row["metrics"]["predicted_p2_information_value"]
            for row in predicted_valid
            if row["metrics"]["predicted_p2_information_value"] is not None
        ),
        "prediction_capture_ratio": stats(
            row["metrics"]["prediction_capture_ratio"]
            for row in predicted_valid
            if row["metrics"]["prediction_capture_ratio"] is not None
        ),
        "prediction_regret_to_perfect_pct": stats(
            100.0 * row["metrics"]["prediction_regret_to_perfect"]
            for row in predicted_valid
            if row["metrics"]["prediction_regret_to_perfect"] is not None
        ),
        "forecast_remote_relative_l1": stats(
            row["metrics"]["forecast_remote_relative_l1"]
            for row in rows
            if row["metrics"].get("forecast_remote_relative_l1") is not None
        ),
        "forecast_rank_pressure_relative_l1": stats(
            row["metrics"]["forecast_rank_pressure_relative_l1"]
            for row in rows
            if row["metrics"].get("forecast_rank_pressure_relative_l1") is not None
        ),
        "reactive_planning_runtime_ms": stats(row["reactive"]["planning_runtime_ms"] for row in valid),
        "perfect_planning_runtime_ms": stats(row["perfect"]["planning_runtime_ms"] for row in valid),
    }


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# RouterSense Scheduling and Prediction Closure",
        "",
        f"Traffic package: `{payload['config']['traffic_package']}`",
        f"Instances: `{payload['config']['selected_instance_count']}`",
        "",
        "## Family summary",
        "",
        "| Family | P01 median | P012 median | Perfect-P2 vs reactive median | Predicted median | Joint planner p50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family_id, summary in payload["family_summary"].items():
        def med(key: str):
            value = summary[key]["median"]
            return "n/a" if value is None else f"{value:.3f}%"
        planner = summary["joint_p012_planning_ms"]["median"]
        lines.append(
            f"| {family_id} | {med('p01_joint_gain_pct')} | {med('p012_joint_gain_pct')} | "
            f"{med('perfect_p2_value_vs_reactive_pct')} | {med('predicted_p2_value_vs_reactive_pct')} | "
            f"{'n/a' if planner is None else f'{planner:.3f} ms'} |"
        )
    exact = payload["exact_summary"]
    lines += [
        "",
        "## Exact information ladder",
        "",
        f"- Valid reactive/perfect controls: `{exact['valid_reactive_perfect_count']}`",
        f"- Perfect P2 value median: `{exact['perfect_p2_information_value_pct']['median']}` percent",
        f"- Predicted P2 value median: `{exact['predicted_p2_information_value_pct']['median']}` percent",
        f"- Prediction capture median: `{exact['prediction_capture_ratio']['median']}`",
        f"- Prediction regret median: `{exact['prediction_regret_to_perfect_pct']['median']}` percent",
        "",
        "Exact reactive is a rolling, non-clairvoyant policy: each replan is exact for currently available information, while O-Joint(P012-perfect) is the clairvoyant certified upper bound for the reduced atomic-edge model.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    records = load_traffic_instances(
        args.traffic_package,
        split=args.split,
        world_sizes=args.world_sizes,
    )
    records = _select(records, max_per_world=max(0, int(args.max_per_world)))
    forecast_map = _load_forecast_map(args.forecast_json)
    forecasts = {
        record.traffic_instance_id: _forecast_for(
            record,
            mode=args.forecast_mode,
            forecast_map=forecast_map,
            corruption_rate=float(args.corruption_rate),
        )
        for record in records
    }
    tasks = [
        (record, family_id, forecasts[record.traffic_instance_id], float(args.prediction_confidence))
        for record in records
        for family_id in args.families
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        family_records = list(executor.map(_family_task, tasks, chunksize=1))
    invalid = [row for row in family_records if not row.valid]
    if invalid:
        raise RuntimeError(f"invalid family closure records: {len(invalid)}")

    exact_candidates = [record for record in records if record.p2_available]
    count = min(max(0, int(args.exact_instance_count)), len(exact_candidates))
    if count:
        indexes = sorted({round(index * (len(exact_candidates) - 1) / max(count - 1, 1)) for index in range(count)})
        exact_selected = [exact_candidates[index] for index in indexes]
    else:
        exact_selected = []
    exact_rows = []
    for record in exact_selected:
        ladder = evaluate_exact_closure(
            record,
            p2_forecast=forecasts[record.traffic_instance_id],
            time_limit_ms=int(args.exact_time_limit_ms),
        )
        exact_rows.append({
            "traffic_instance_id": record.traffic_instance_id,
            "model_id": record.model_id,
            "world_size": record.world_size,
            "layer_id": record.layer_id,
            **ladder.to_dict(include_schedule=False),
        })

    payload = {
        "schema_version": "scheduling_prediction_closure.v1",
        "config": {
            "traffic_package": str(args.traffic_package),
            "split": args.split,
            "world_sizes": list(args.world_sizes),
            "families": list(args.families),
            "selected_instance_count": len(records),
            "forecast_mode": args.forecast_mode,
            "corruption_rate": float(args.corruption_rate),
            "prediction_confidence": float(args.prediction_confidence),
            "exact_instance_count": len(exact_rows),
        },
        "family_summary": summarize_family_records(family_records),
        "exact_summary": _exact_summary(exact_rows),
        "family_records": [row.to_dict() for row in family_records] if args.include_records else [],
        "exact_records": exact_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "closure_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "closure_report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps({"family_summary": payload["family_summary"], "exact_summary": payload["exact_summary"]}, indent=2))


if __name__ == "__main__":
    main()
