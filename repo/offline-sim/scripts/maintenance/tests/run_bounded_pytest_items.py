#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "RS_SIM_BOUNDED_PYTEST"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index", "nodeid", "test_file", "status", "return_code", "timed_out",
        "item_duration_seconds", "process_elapsed_seconds", "log_path",
    ]
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temp.replace(path)


def _kill_process_group(proc: subprocess.Popen[Any], grace_seconds: float) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _collect(root: str, output: Path, timeout_seconds: float) -> list[str]:
    command = [
        sys.executable,
        "tools/collect_pytest_nodeids.py",
        root,
        "--output",
        str(output),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    deadline = time.monotonic() + timeout_seconds
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.poll() is None:
        _kill_process_group(proc, 2.0)
        raise RuntimeError(f"pytest collection exceeded {timeout_seconds:g}s")
    if proc.returncode != 0:
        raise RuntimeError(f"pytest collection failed with return code {proc.returncode}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    return [str(item) for item in payload["nodeids"]]


def _run_item(
    *,
    index: int,
    nodeid: str,
    output_dir: Path,
    item_timeout_seconds: float,
    kill_grace_seconds: float,
) -> dict[str, Any]:
    slug = f"{index:04d}"
    result_path = output_dir / "item_results" / f"{slug}.json"
    log_path = output_dir / "item_logs" / f"{slug}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["RS_SIM_PYTEST_ITEM_TIMEOUT_SECONDS"] = str(item_timeout_seconds)
    command = [
        sys.executable,
        "tools/pytest_item_worker.py",
        nodeid,
        "--result",
        str(result_path),
    ]
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            start_new_session=True,
        )
        # Allow a small reporting margin after the in-pytest five-second alarm.
        process_deadline = started + item_timeout_seconds + kill_grace_seconds
        while proc.poll() is None and time.monotonic() < process_deadline:
            time.sleep(0.02)
        if proc.poll() is None:
            timed_out = True
            _kill_process_group(proc, kill_grace_seconds)
    elapsed = time.monotonic() - started
    return_code = int(proc.returncode) if proc.poll() is not None else -999
    item_payload: dict[str, Any] = {}
    if result_path.is_file():
        try:
            item_payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            item_payload = {}
    duration = float(item_payload.get("item_duration_seconds", item_timeout_seconds if timed_out else elapsed))
    timed_out = bool(timed_out or duration >= item_timeout_seconds)
    status = "TIMEOUT" if timed_out else ("PASS" if return_code == 0 else "FAIL")
    return {
        "index": int(index),
        "nodeid": nodeid,
        "test_file": nodeid.split("::", 1)[0],
        "status": status,
        "return_code": return_code,
        "timed_out": timed_out,
        "item_duration_seconds": round(duration, 6),
        "process_elapsed_seconds": round(elapsed, 6),
        "phase_durations_seconds": item_payload.get("phase_durations_seconds", {}),
        "phase_outcomes": item_payload.get("phase_outcomes", {}),
        "log_path": str(log_path),
        "result_path": str(result_path) if result_path.is_file() else None,
    }


def _summary(
    *,
    root: str,
    all_nodeids: list[str],
    selected_start: int,
    selected: list[str],
    rows: list[dict[str, Any]],
    item_limit: float,
    file_limit: float,
    in_progress: bool,
) -> dict[str, Any]:
    file_totals: dict[str, float] = {}
    for row in rows:
        file_totals[row["test_file"]] = file_totals.get(row["test_file"], 0.0) + float(row["item_duration_seconds"])
    file_rows = [
        {
            "test_file": name,
            "item_duration_seconds_sum": round(total, 6),
            "within_limit": total <= file_limit,
        }
        for name, total in sorted(file_totals.items())
    ]
    violations = [row for row in file_rows if not row["within_limit"]]
    return {
        "schema_version": SCHEMA,
        "root": root,
        "in_progress": in_progress,
        "collected_item_count": len(all_nodeids),
        "selected_start": selected_start,
        "selected_item_count": len(selected),
        "completed_item_count": len(rows),
        "passed_item_count": sum(row["status"] == "PASS" for row in rows),
        "failed_item_count": sum(row["status"] == "FAIL" for row in rows),
        "timed_out_item_count": sum(row["status"] == "TIMEOUT" for row in rows),
        "item_timeout_seconds": item_limit,
        "file_item_time_limit_seconds": file_limit,
        "file_limit_violation_count": len(violations),
        "file_aggregates": file_rows,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pytest items sequentially in fresh bounded processes")
    parser.add_argument("root", nargs="?", default="tests")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--item-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--file-time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=2.0)
    parser.add_argument("--collection-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bounded_pytest"))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.start < 0 or (args.limit is not None and args.limit < 0):
        raise SystemExit("--start and --limit must be non-negative")
    if min(args.item_timeout_seconds, args.file_time_limit_seconds, args.kill_grace_seconds) <= 0:
        raise SystemExit("timeout and limit values must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    nodeids_path = output_dir / "collected_nodeids.json"
    all_nodeids = _collect(args.root, nodeids_path, args.collection_timeout_seconds)
    stop = None if args.limit is None else args.start + args.limit
    selected = all_nodeids[args.start:stop]
    rows: list[dict[str, Any]] = []
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "timings.csv"

    for offset, nodeid in enumerate(selected):
        index = args.start + offset
        row = _run_item(
            index=index,
            nodeid=nodeid,
            output_dir=output_dir,
            item_timeout_seconds=args.item_timeout_seconds,
            kill_grace_seconds=args.kill_grace_seconds,
        )
        rows.append(row)
        summary = _summary(
            root=args.root,
            all_nodeids=all_nodeids,
            selected_start=args.start,
            selected=selected,
            rows=rows,
            item_limit=args.item_timeout_seconds,
            file_limit=args.file_time_limit_seconds,
            in_progress=True,
        )
        _atomic_json(summary_path, summary)
        _write_csv(csv_path, rows)
        print(
            f"[{index + 1}/{len(all_nodeids)}] {row['status']:<7} "
            f"item={row['item_duration_seconds']:.3f}s process={row['process_elapsed_seconds']:.3f}s {nodeid}",
            flush=True,
        )
        if args.fail_fast and row["status"] != "PASS":
            break

    summary = _summary(
        root=args.root,
        all_nodeids=all_nodeids,
        selected_start=args.start,
        selected=selected,
        rows=rows,
        item_limit=args.item_timeout_seconds,
        file_limit=args.file_time_limit_seconds,
        in_progress=False,
    )
    _atomic_json(summary_path, summary)
    _write_csv(csv_path, rows)
    compact = {k: v for k, v in summary.items() if k not in {"rows", "file_aggregates"}}
    print(json.dumps(compact, indent=2, sort_keys=True))
    clean = (
        len(rows) == len(selected)
        and summary["failed_item_count"] == 0
        and summary["timed_out_item_count"] == 0
        and summary["file_limit_violation_count"] == 0
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
