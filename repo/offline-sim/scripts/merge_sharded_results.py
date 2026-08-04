from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while limit > 131072:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _status(path: Path) -> dict[str, Any]:
    progress = path.with_suffix(path.suffix + ".progress.json")
    if not progress.is_file():
        return {"status": "NO_PROGRESS"}
    return json.loads(progress.read_text(encoding="utf-8"))


def main() -> int:
    _raise_csv_field_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shard_files = sorted(args.input_dir.glob("main_measured_1m_shard_*.csv"))
    raw_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    shard_summary: list[dict[str, Any]] = []

    for path in shard_files:
        rows = _read_rows(path)
        status = _status(path)
        for row in rows:
            row = dict(row)
            row["merge__source_csv"] = path.name
            raw_rows.append(row)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        failures = [
            row for row in rows
            if row.get("description__record_type") in {"FAILURE", "TIMEOUT"}
            or row.get("description__status") in {"FAILED", "TIMEOUT"}
        ]
        trace_pass = [
            row for row in rows
            if row.get("description__record_type") == "TRACE_COMPLETE"
            and row.get("description__status") == "PASS"
        ]
        shard_summary.append(
            {
                "source_csv": path.name,
                "status": status.get("status", "NO_PROGRESS"),
                "completed_runs": int(status.get("completed_runs", 0) or 0),
                "failed_runs": int(status.get("failed_runs", 0) or 0),
                "processed_fixtures": int(status.get("processed_fixtures", 0) or 0),
                "fixture_count": int(status.get("fixture_count", 0) or 0),
                "csv_rows": len(rows),
                "failure_rows": len(failures),
                "trace_complete_pass_rows": len(trace_pass),
            }
        )

    by_run_key: OrderedDict[str, dict[str, str]] = OrderedDict()
    for row in raw_rows:
        run_key = row.get("description__run_key") or ""
        if not run_key:
            run_key = f"__row_{len(by_run_key)}"
        by_run_key[run_key] = row
    latest_rows = list(by_run_key.values())

    failure_rows = [
        row for row in raw_rows
        if row.get("description__record_type") in {"FAILURE", "TIMEOUT"}
        or row.get("description__status") in {"FAILED", "TIMEOUT"}
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "main_measured_1m_raw_merged.csv", raw_rows, fieldnames)
    _write_rows(args.output_dir / "main_measured_1m_latest_by_run_key.csv", latest_rows, fieldnames)
    _write_rows(args.output_dir / "main_measured_1m_failures.csv", failure_rows, fieldnames)

    summary_fields = [
        "source_csv",
        "status",
        "completed_runs",
        "failed_runs",
        "processed_fixtures",
        "fixture_count",
        "csv_rows",
        "failure_rows",
        "trace_complete_pass_rows",
    ]
    _write_rows(
        args.output_dir / "main_measured_1m_shard_summary.csv",
        [{k: str(v) for k, v in row.items()} for row in shard_summary],
        summary_fields,
    )

    pass_shards = [row for row in shard_summary if row["status"] == "PASS"]
    report = {
        "input_dir": str(args.input_dir),
        "shard_csv_count": len(shard_files),
        "raw_rows": len(raw_rows),
        "latest_by_run_key_rows": len(latest_rows),
        "failure_rows": len(failure_rows),
        "pass_shards": len(pass_shards),
        "non_pass_shards": len(shard_summary) - len(pass_shards),
        "completed_runs_progress_sum": sum(row["completed_runs"] for row in shard_summary),
        "failed_runs_progress_sum": sum(row["failed_runs"] for row in shard_summary),
        "processed_fixtures_progress_sum": sum(row["processed_fixtures"] for row in shard_summary),
        "trace_complete_pass_rows": sum(row["trace_complete_pass_rows"] for row in shard_summary),
        "outputs": {
            "raw_merged": "main_measured_1m_raw_merged.csv",
            "latest_by_run_key": "main_measured_1m_latest_by_run_key.csv",
            "failures": "main_measured_1m_failures.csv",
            "shard_summary": "main_measured_1m_shard_summary.csv",
        },
    }
    (args.output_dir / "MERGE_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
