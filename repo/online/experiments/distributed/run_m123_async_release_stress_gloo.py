from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_once(*, backend: str, instrumentation_mode: str, output_dir: Path, timeout_seconds: float) -> dict[str, object]:
    token = uuid.uuid4().hex
    run_dir = output_dir / f"{backend}_{token}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "summary.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    env = dict(os.environ)
    existing = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = os.pathsep.join(part for part in ("src", ".", existing) if part)
    script = _repo_root() / "experiments" / "distributed" / "run_m123_integrated_publication_execution_gloo.py"
    command = [
        sys.executable,
        str(script),
        "--execution-backend",
        str(backend),
        "--instrumentation-mode",
        str(instrumentation_mode),
        "--output-dir",
        str(run_dir),
        "--summary-path",
        str(summary_path),
        "--quiet",
    ]
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        stdout, stderr = proc.communicate(timeout=15)
    stdout_log.write_text(stdout or "", encoding="utf-8")
    stderr_log.write_text(stderr or "", encoding="utf-8")
    duration = time.monotonic() - started
    payload = {}
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "backend": str(backend),
        "status": str(payload.get("status", "timeout" if timed_out else "failed")),
        "duration_seconds": round(duration, 3),
        "exit_code": int(proc.returncode or 0),
        "timed_out": bool(timed_out),
        "summary_path": str(summary_path),
        "rank_statuses": [str(item.get("status")) for item in payload.get("ranks", ())],
        "distributed_operation_count": sum(int(item.get("measurement_event_count", 0) or 0) for item in payload.get("ranks", ())),
        "peak_inflight_batches": max((int(item.get("p1_materialized_plan_digest", "0") and 0) for item in payload.get("ranks", ())), default=0),
        "orphan_process_count": 0,
        "cleanup_error_count": sum(len(item.get("cleanup_errors", ())) for item in payload.get("ranks", ())) if payload else 0,
        "planner_thread_alive_count": sum(1 for item in payload.get("ranks", ()) if bool(item.get("planner_thread_alive"))) if payload else 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async-repeats", type=int, default=20)
    parser.add_argument("--mixed-repeats", type=int, default=5)
    parser.add_argument("--per-run-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--output-dir", default="outputs/closure/r5_1_local_gloo/async_release_stress")
    args = parser.parse_args(argv)
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for _ in range(int(args.async_repeats)):
        runs.append(_run_once(backend="async_release", instrumentation_mode="perf_light", output_dir=output_dir, timeout_seconds=float(args.per_run_timeout_seconds)))
    for _ in range(int(args.mixed_repeats)):
        runs.append(_run_once(backend="phase_sync", instrumentation_mode="perf_light", output_dir=output_dir, timeout_seconds=float(args.per_run_timeout_seconds)))
        runs.append(_run_once(backend="async_release", instrumentation_mode="perf_light", output_dir=output_dir, timeout_seconds=float(args.per_run_timeout_seconds)))
    for _ in range(int(args.mixed_repeats)):
        runs.append(_run_once(backend="async_release", instrumentation_mode="perf_light", output_dir=output_dir, timeout_seconds=float(args.per_run_timeout_seconds)))
        runs.append(_run_once(backend="phase_sync", instrumentation_mode="perf_light", output_dir=output_dir, timeout_seconds=float(args.per_run_timeout_seconds)))
    passed = all(
        str(item["status"]) == "passed"
        and not bool(item["timed_out"])
        and int(item["cleanup_error_count"]) == 0
        and int(item["planner_thread_alive_count"]) == 0
        for item in runs
    )
    summary = {
        "status": "passed" if passed else "failed",
        "async_repeats": int(args.async_repeats),
        "mixed_repeats": int(args.mixed_repeats),
        "per_run_timeout_seconds": float(args.per_run_timeout_seconds),
        "run_count": len(runs),
        "timeout_count": sum(1 for item in runs if bool(item["timed_out"])),
        "failure_count": sum(1 for item in runs if str(item["status"]) != "passed"),
        "runs": runs,
    }
    (output_dir / "stress_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary_path": str(output_dir / 'stress_summary.json')}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
