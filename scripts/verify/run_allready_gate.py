#!/usr/bin/env python3
"""Run the recoverable RouterSense code/deployment readiness gate.

The gate deliberately executes each default pytest file in a fresh process.
This prevents one leaked distributed/subprocess resource from hiding the
results of every later test and produces a durable per-file audit trail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "allready"
DEFAULT_TRACE_ROOT = ROOT / "external_traces"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def base_env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TORCH_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return env


def safe_name(path: Path | str) -> str:
    return str(path).replace("/", "__").replace("\\", "__").replace(":", "_")


def display_path(path: Path) -> str:
    """Return a stable report path for both in-repo and external outputs."""

    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def _stop_process_tree(process: subprocess.Popen[Any]) -> None:
    """Stop a timed-out command and reap its process group deterministically."""

    if process.poll() is None:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - exercised by Windows deployment hosts
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover - exercised by Windows deployment hosts
                    process.kill()
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - pathological kernel state
        pass


def run_command(
    argv: list[str],
    *,
    log_path: Path,
    timeout_seconds: int,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        popen_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "env": env or base_env(),
            "text": True,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        else:  # pragma: no cover - exercised by Windows deployment hosts
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(argv, **popen_kwargs)
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_tree(process)
            log_file.write("\n[TIMEOUT]\n")
            log_file.flush()
    output = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "command": argv,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "duration_seconds": round(time.perf_counter() - start, 6),
        "log": display_path(log_path),
        "tail": output[-2000:],
    }


def git_value(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def verify_checksum_manifest(manifest: Path) -> dict[str, Any]:
    root = manifest.parent
    failures: list[dict[str, str]] = []
    checked = 0
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*").strip()
        target = root / relative
        checked += 1
        if not target.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected:
            failures.append({"path": relative, "reason": "sha256_mismatch", "actual": digest, "expected": expected})
    return {
        "manifest": display_path(manifest),
        "checked": checked,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def iter_default_test_files() -> list[Path]:
    result = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        if "legacy" in path.relative_to(ROOT / "tests").parts:
            continue
        result.append(path)
    return result


def segmented_pytest(*, output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    log_root = output_dir / "logs" / "pytest_files"
    for index, path in enumerate(iter_default_test_files(), start=1):
        relative = path.relative_to(ROOT)
        result = run_command(
            [sys.executable, "-m", "pytest", "-q", str(relative)],
            log_path=log_root / f"{index:03d}_{safe_name(relative)}.log",
            timeout_seconds=timeout_seconds,
        )
        if result["timed_out"]:
            status = "TIMEOUT"
        elif result["returncode"] == 0:
            status = "PASS"
        elif result["returncode"] == 5:
            status = "DESELECTED"
        else:
            status = "FAIL"
        rows.append({"test_file": str(relative), "status": status, **result})
        print(f"[pytest {index:03d}/{len(iter_default_test_files()):03d}] {status:10s} {relative}", flush=True)
    counts = {name: sum(row["status"] == name for row in rows) for name in ("PASS", "DESELECTED", "FAIL", "TIMEOUT")}
    return {
        "status": "PASS" if counts["FAIL"] == 0 and counts["TIMEOUT"] == 0 else "FAIL",
        "counts": counts,
        "files": rows,
    }


def compiled_kernel_warmup(*, output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    probe = """
import json
from rs.runtime.online.megatron_ep.target_planning.planner_service import TargetLayerPlannerService
from rs.runtime.online.megatron_ep.target_planning.store import TargetPlanStore
service = TargetLayerPlannerService(store=TargetPlanStore())
service.start()
try:
    rows = [row for row in service.timeline() if row.get('event') == 'compiled_kernel_warmup']
    if not rows or rows[-1].get('status') != 'passed':
        raise RuntimeError(f'compiled warmup timeline missing or failed: {rows!r}')
    print(json.dumps(rows[-1], sort_keys=True))
finally:
    service.close()
"""
    result = run_command(
        [sys.executable, "-c", probe],
        log_path=output_dir / "logs" / "compiled_kernel_warmup.log",
        timeout_seconds=max(timeout_seconds, 300),
    )
    payload: dict[str, Any] = {}
    if result["returncode"] == 0 and not result["timed_out"]:
        try:
            payload = json.loads(
                (output_dir / "logs" / "compiled_kernel_warmup.log")
                .read_text(encoding="utf-8", errors="replace")
                .strip()
                .splitlines()[-1]
            )
        except Exception as exc:
            payload = {"parse_error": f"{type(exc).__name__}: {exc}"}
    passed = (
        result["returncode"] == 0
        and not result["timed_out"]
        and payload.get("status") == "passed"
        and tuple(payload.get("planner_ids", ()))
        == (
            "future:p012:joint:event:rscf",
            "future:p012:joint:global:rscf",
        )
    )
    return {"status": "PASS" if passed else "FAIL", "snapshot": payload, **result}


def explicit_gloo_tests(*, output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    candidates: list[Path] = []
    for path in iter_default_test_files():
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if "gloo" in text or "init_process_group" in text or "torch.distributed" in text:
            candidates.append(path)
    if not candidates:
        return {"status": "BLOCKED", "reason": "no explicit Gloo test files found", "files": []}
    result = run_command(
        [sys.executable, "-m", "pytest", "-q", *[str(path.relative_to(ROOT)) for path in candidates]],
        log_path=output_dir / "logs" / "gloo_explicit.log",
        timeout_seconds=max(timeout_seconds, 300),
    )
    status = "PASS" if result["returncode"] in {0, 5} and not result["timed_out"] else "FAIL"
    return {"status": status, "files": [str(path.relative_to(ROOT)) for path in candidates], **result}


def trace_mode_matrix(*, output_dir: Path, trace_root: Path, timeout_seconds: int) -> dict[str, Any]:
    rows = []
    for core in ("gmwd", "rsbc", "rscf"):
        for branch in ("event", "global"):
            report = output_dir / "reports" / f"trace_modes_{core}_{branch}.json"
            result = run_command(
                [
                    sys.executable,
                    "-m",
                    "experiments.offline.compare_p012_p0123_future",
                    "--bundle",
                    str(trace_root),
                    "--output",
                    str(report),
                    "--core",
                    core,
                    "--branch",
                    branch,
                    "--split-prefix",
                    "val-",
                    "--p3-weight",
                    "0.01",
                ],
                log_path=output_dir / "logs" / f"trace_modes_{core}_{branch}.log",
                timeout_seconds=max(timeout_seconds, 600),
            )
            status = "PASS" if result["returncode"] == 0 and not result["timed_out"] and report.is_file() else "FAIL"
            aggregate: dict[str, Any] = {}
            if report.is_file():
                try:
                    aggregate = json.loads(report.read_text(encoding="utf-8")).get("aggregate", {})
                except Exception as exc:  # pragma: no cover - report corruption path
                    aggregate = {"parse_error": f"{type(exc).__name__}: {exc}"}
                    status = "FAIL"
            rows.append({"core": core, "branch": branch, "status": status, "report": display_path(report), "aggregate": aggregate, **result})
            print(f"[trace] {status:4s} {core}/{branch}", flush=True)
    return {"status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL", "rows": rows}


def deploy_gate(*, output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    shell_files = sorted((ROOT / "scripts" / "deploy").glob("*.sh")) + sorted((ROOT / "deploy" / "remote").glob("*.sh"))
    syntax = run_command(
        ["bash", "-n", *[str(path.relative_to(ROOT)) for path in shell_files]],
        log_path=output_dir / "logs" / "deploy_shell_syntax.log",
        timeout_seconds=timeout_seconds,
    )
    commands = {
        "access_dry_run": ["bash", "scripts/deploy/verify_cluster_access.sh", "deploy/inventory/hosts.example.yaml"],
        "launch_dry_run": ["bash", "scripts/deploy/launch_remote.sh", "deploy/inventory/hosts.example.yaml"],
        "sync_dry_run": ["bash", "scripts/deploy/sync_repo.sh", "deploy/inventory/hosts.example.yaml"],
        "prepare_dry_run": ["bash", "scripts/deploy/prepare_cluster_environment.sh", "deploy/inventory/hosts.example.yaml"],
        "parity_dry_run": ["bash", "scripts/deploy/verify_repo_parity.sh", "deploy/inventory/hosts.example.yaml"],
        "model_sync_dry_run": ["bash", "scripts/deploy/sync_model_cache.sh", "deploy/inventory/hosts.example.yaml"],
        "collect_dry_run": ["bash", "scripts/deploy/collect_remote_logs.sh", "deploy/inventory/hosts.example.yaml", "--run-id", "allready-gate"],
        "stop_dry_run": ["bash", "scripts/deploy/stop_remote_jobs.sh", "deploy/inventory/hosts.example.yaml", "--run-id", "allready-gate"],
        "pipeline_dry_run": ["bash", "scripts/deploy/run_allready_pipeline.sh", "deploy/inventory/hosts.example.yaml", "--run-id", "allready-gate"],
        "strategy_dry_run": [
            sys.executable,
            "-m",
            "experiments.online.run_deployed_strategy",
            "--comparison-config",
            "configs/official/online_p012_deploy_smoke.yaml",
            "--strategy",
            "routersense_future_p012_joint_global_rscf_async",
            "--output-dir",
            str(output_dir / "strategy_dry_run"),
            "--run-id",
            "allready-gate",
            "--model-path",
            "/tmp/router-sense-model-placeholder",
            "--dry-run",
        ],
    }
    results: dict[str, Any] = {}
    for name, argv in commands.items():
        results[name] = run_command(
            argv,
            log_path=output_dir / "logs" / f"deploy_{name}.log",
            timeout_seconds=max(timeout_seconds, 180),
        )
    expected_fragments = {
        "access_dry_run": '"status": "DRY_RUN"',
        "launch_dry_run": '"status": "DRY_RUN"',
        "sync_dry_run": '"status": "DRY_RUN"',
        "prepare_dry_run": '"status": "DRY_RUN"',
        "parity_dry_run": '"verification_status": "DRY_RUN"',
        "model_sync_dry_run": '"status": "DRY_RUN"',
        "collect_dry_run": '"status": "DRY_RUN"',
        "stop_dry_run": '"status": "DRY_RUN"',
        "pipeline_dry_run": '"status": "DRY_RUN_PASS"',
        "strategy_dry_run": '"dry_run": true',
    }
    checks = {
        name: result["returncode"] == 0
        and not result["timed_out"]
        and expected_fragments[name] in result["tail"]
        for name, result in results.items()
    }
    syntax_ok = syntax["returncode"] == 0 and not syntax["timed_out"]
    return {
        "status": "PASS" if syntax_ok and all(checks.values()) else "FAIL",
        "shell_syntax": syntax,
        "checks": checks,
        **results,
    }


def environment_probe(*, output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    gpu = run_command(
        [sys.executable, "-m", "scripts.verify.verify_gpu_environment"],
        log_path=output_dir / "logs" / "gpu_environment.log",
        timeout_seconds=timeout_seconds,
    )
    payload: dict[str, Any] = {}
    try:
        payload = json.loads((ROOT / gpu["log"]).read_text(encoding="utf-8")) if not Path(gpu["log"]).is_absolute() else json.loads(Path(gpu["log"]).read_text(encoding="utf-8"))
    except Exception:
        payload = {"raw_tail": gpu["tail"]}
    cuda_available = bool(payload.get("cuda_available", False))
    visible = int(payload.get("visible_gpu_count", payload.get("device_count", 0)) or 0)
    return {
        "status": "READY" if cuda_available and visible >= 1 else "BLOCKED",
        "reason": None if cuda_available and visible >= 1 else "CPU-only validation host; CUDA/NCCL and real multi-node execution remain unverified",
        "gpu_probe": gpu,
        "snapshot": payload,
    }


def render_markdown(report: dict[str, Any]) -> str:
    tests = report["stages"]["segmented_default_pytest"]
    trace = report["stages"]["trace_mode_matrix"]
    env = report["stages"]["environment"]
    lines = [
        "# RouterSense All-Ready Validation",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Commit: `{report['commit']}`",
        f"- Overall code/deploy gate: **{report['overall_status']}**",
        f"- Hardware execution readiness on this host: **{env['status']}**",
        "",
        "## Required gates",
        "",
        f"- Compile: **{report['stages']['compileall']['status']}**",
        f"- Empty-cache compiled-kernel warmup: **{report['stages']['compiled_kernel_warmup']['status']}**",
        f"- Segmented default pytest: **{tests['status']}** — {tests['counts']}",
        f"- Explicit Gloo tests: **{report['stages']['gloo']['status']}**",
        f"- Trace checksums: **{report['stages']['trace_checksums']['status']}**",
        f"- P012/P0123/Future trace matrix: **{trace['status']}**",
        f"- Deployment syntax and dry-run: **{report['stages']['deploy']['status']}**",
        "",
        "## Trace-mode matrix",
        "",
        "| Core | Branch | Status | Instances | Future equivalence | P0123 W/T/L |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in trace.get("rows", []):
        agg = row.get("aggregate", {})
        lines.append(
            f"| {row['core']} | {row['branch']} | {row['status']} | {agg.get('instances', '')} | "
            f"{agg.get('future_plan_equivalence_rate', '')} | {agg.get('p0123_vs_p012_wins_ties_losses', '')} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            env.get("reason") or "GPU environment is visible.",
            "A PASS here means the source, CPU contracts, trace assets, Gloo path, and deployment dry-run are ready. "
            "It does not fabricate CUDA/NCCL or physical two-node evidence when those resources are absent.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--per-file-timeout", type=int, default=180)
    parser.add_argument("--skip-trace-modes", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    trace_root = args.trace_root.resolve()
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    numba_cache = output_dir / "numba_cache"
    shutil.rmtree(numba_cache, ignore_errors=True)
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)

    report: dict[str, Any] = {
        "schema_version": "routersense.allready.gate.v2",
        "generated_at": utc_now(),
        "repo_root": str(ROOT),
        "commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "python": sys.version,
        "stages": {},
    }

    compile_result = run_command(
        [sys.executable, "-m", "compileall", "-q", "src", "experiments", "scripts", "tests"],
        log_path=output_dir / "logs" / "compileall.log",
        timeout_seconds=max(args.per_file_timeout, 300),
    )
    report["stages"]["compileall"] = {"status": "PASS" if compile_result["returncode"] == 0 and not compile_result["timed_out"] else "FAIL", **compile_result}

    report["stages"]["compiled_kernel_warmup"] = compiled_kernel_warmup(
        output_dir=output_dir, timeout_seconds=args.per_file_timeout
    )
    report["stages"]["segmented_default_pytest"] = segmented_pytest(output_dir=output_dir, timeout_seconds=args.per_file_timeout)
    report["stages"]["gloo"] = explicit_gloo_tests(output_dir=output_dir, timeout_seconds=args.per_file_timeout)

    manifests = sorted(trace_root.rglob("checksums.sha256")) if trace_root.exists() else []
    checksum_rows = [verify_checksum_manifest(path) for path in manifests]
    if args.skip_trace_modes and not checksum_rows:
        report["stages"]["trace_checksums"] = {
            "status": "SKIPPED",
            "reason": "explicit --skip-trace-modes and no external trace bundle was supplied",
            "manifest_count": 0,
            "manifests": [],
        }
    else:
        report["stages"]["trace_checksums"] = {
            "status": "PASS" if checksum_rows and all(row["status"] == "PASS" for row in checksum_rows) else "FAIL",
            "manifest_count": len(checksum_rows),
            "manifests": checksum_rows,
        }

    if args.skip_trace_modes:
        report["stages"]["trace_mode_matrix"] = {
            "status": "SKIPPED",
            "reason": "explicit --skip-trace-modes; use --trace-root with the external validation bundle for the full matrix",
            "rows": [],
        }
    else:
        report["stages"]["trace_mode_matrix"] = trace_mode_matrix(
            output_dir=output_dir,
            trace_root=trace_root,
            timeout_seconds=args.per_file_timeout,
        )
    report["stages"]["deploy"] = deploy_gate(output_dir=output_dir, timeout_seconds=args.per_file_timeout)
    report["stages"]["environment"] = environment_probe(output_dir=output_dir, timeout_seconds=args.per_file_timeout)

    required = (
        "compileall",
        "compiled_kernel_warmup",
        "segmented_default_pytest",
        "gloo",
        "trace_checksums",
        "trace_mode_matrix",
        "deploy",
    )
    report["overall_status"] = "PASS" if all(report["stages"][name]["status"] in {"PASS", "SKIPPED"} for name in required) else "FAIL"
    json_path = output_dir / "reports" / "allready_gate.json"
    md_path = output_dir / "reports" / "ALL_READY_VALIDATION.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"], "report": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
