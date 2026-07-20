#!/usr/bin/env python3
from __future__ import annotations

"""Run the complete RouterSense deployment preparation and launch pipeline."""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.topology import DEFAULT_DEPLOYMENT_MODEL_ID


DEFAULT_CONFIG = "configs/official/online_p012_deploy_smoke.yaml"
DEFAULT_STRATEGY = "routersense_future_p012_joint_global_rscf_async"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tail_text(path: Path, *, line_count: int = 80) -> str:
    if not path.is_file():
        return "<log file missing>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(int(line_count), 1):])


def _write_human_summary(output_root: Path, report: dict[str, Any]) -> Path:
    """Write the one file an execution-only agent should read first."""

    status = str(report.get("status", "UNKNOWN"))
    failed = [row for row in report.get("stages", []) if str(row.get("status")) != "PASS"]
    filename = "failure_summary.txt" if status == "FAIL" else "run_summary.txt"
    path = output_root / filename
    lines = [
        f"RouterSense deployment pipeline: {status}",
        f"run_id: {report.get('run_id')}",
        f"generated_at: {report.get('generated_at')}",
        f"pipeline_report: {output_root / 'pipeline_report.json'}",
        "",
    ]
    if status == "FAIL":
        lines.extend(
            [
                "STOP. Do not edit Python, planner parameters, experiment YAML, or result JSON.",
                "Return this file, pipeline_report.json, the complete logs directory, and collected deployment artifacts.",
                "",
                "Failed or incomplete stages:",
            ]
        )
        if not failed:
            lines.append("- pipeline did not complete every required stage")
        for row in failed:
            log_rel = str(row.get("log", ""))
            log_path = ROOT / log_rel if log_rel else Path()
            lines.extend(
                [
                    f"- stage: {row.get('name')}",
                    f"  returncode: {row.get('returncode')}",
                    f"  timed_out: {row.get('timed_out')}",
                    f"  command: {' '.join(str(x) for x in row.get('command', []))}",
                    f"  log: {log_path if log_rel else '<missing>'}",
                    "  log tail:",
                ]
            )
            lines.extend(f"    {line}" for line in _tail_text(log_path).splitlines())
        collected_root = ROOT / "outputs" / "deployment" / str(report.get("run_id", ""))
        remote_logs = sorted(collected_root.rglob("*.log")) if collected_root.is_dir() else []
        if remote_logs:
            lines.extend(["", "Collected remote log tails:"])
            for remote_log in remote_logs[:8]:
                lines.append(f"--- {remote_log} ---")
                lines.extend(_tail_text(remote_log, line_count=60).splitlines())
    else:
        lines.extend(
            [
                "All required pipeline stages completed successfully.",
                "For an applied run, inspect the collected deployment_result_summary.json before accepting performance data.",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--comparison-config", default=DEFAULT_CONFIG)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--stage-timeout-seconds", type=int, default=1800)
    parser.add_argument("--force-sync", action="store_true")
    parser.add_argument("--include-dev", action="store_true")
    parser.add_argument("--skip-repo-sync", action="store_true")
    parser.add_argument("--skip-model-sync", action="store_true")
    parser.add_argument("--skip-model-preflight", action="store_true")
    parser.add_argument("--skip-runtime-environment-preflight", action="store_true")
    parser.add_argument("--skip-link-calibration", action="store_true")
    parser.add_argument("--link-cost-profile", default=None)
    parser.add_argument("--model-id", default=DEFAULT_DEPLOYMENT_MODEL_ID)
    parser.add_argument("--nccl-socket-ifname", default=os.environ.get("NCCL_SOCKET_IFNAME", ""))
    parser.add_argument("--no-collect", action="store_true")
    return parser.parse_args(argv)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - Windows deployment controller
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
                else:  # pragma: no cover - Windows deployment controller
                    process.kill()
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _run_stage(
    name: str,
    command: list[str],
    *,
    log_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started_at = _utc_now()
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        kwargs: dict[str, Any] = {
            "cwd": str(ROOT),
            "env": os.environ,
            "text": True,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
        }
        if os.name != "nt":
            kwargs["start_new_session"] = True
        else:  # pragma: no cover - Windows deployment controller
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(command, **kwargs)
        try:
            process.wait(timeout=max(int(timeout_seconds), 1))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            log_file.write("\n[TIMEOUT]\n")
    output = log_path.read_text(encoding="utf-8", errors="replace")
    payload: Any = None
    try:
        payload = json.loads(output)
    except Exception:
        payload = {"raw_tail": output[-4000:]}
    return {
        "name": name,
        "command": command,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "status": "PASS" if process.returncode == 0 and not timed_out else "FAIL",
        "payload": payload,
        "log": str(log_path.relative_to(ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.skip_link_calibration and not args.link_cost_profile:
        raise SystemExit("--skip-link-calibration requires --link-cost-profile")
    if args.link_cost_profile and not args.skip_link_calibration:
        raise SystemExit(
            "--link-cost-profile is only valid with --skip-link-calibration; "
            "the normal path calibrates and uses this run's canonical profile"
        )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("rs-%Y%m%dT%H%M%SZ")
    output_root = ROOT / "outputs" / "deployment_pipeline" / run_id
    log_dir = output_root / "logs"
    apply_flag = ["--apply"] if args.apply else []
    stages: list[tuple[str, list[str]]] = [
        ("access", ["bash", "scripts/deploy/verify_cluster_access.sh", args.inventory, *apply_flag]),
    ]
    if not args.skip_repo_sync:
        sync = ["bash", "scripts/deploy/sync_repo.sh", args.inventory, *apply_flag]
        if args.force_sync:
            sync.append("--force")
        stages.extend(
            [
                ("repo_sync", sync),
                ("repo_parity", ["bash", "scripts/deploy/verify_repo_parity.sh", args.inventory, *apply_flag]),
            ]
        )
    prepare = ["bash", "scripts/deploy/prepare_cluster_environment.sh", args.inventory, *apply_flag]
    if args.include_dev:
        prepare.append("--include-dev")
    stages.append(("environment", prepare))
    if not args.skip_model_sync:
        stages.append((
            "model_sync",
            [
                "bash", "scripts/deploy/sync_model_cache.sh", args.inventory,
                "--model-id", args.model_id, "--require-existing", *apply_flag,
            ],
        ))
    if not args.skip_model_preflight:
        stages.append((
            "model_preflight",
            [
                "bash", "scripts/deploy/verify_mounted_model.sh", args.inventory,
                "--model-id", args.model_id, *apply_flag,
            ],
        ))
    if not args.skip_runtime_environment_preflight:
        stages.append((
            "runtime_environment_preflight",
            [
                "bash", "scripts/deploy/verify_runtime_environment.sh", args.inventory,
                "--model-id", args.model_id, *apply_flag,
            ],
        ))
    profile_relative = (
        str(args.link_cost_profile)
        if args.skip_link_calibration
        else f"outputs/deployment_profiles/{run_id}/link_cost_profile.json"
    )
    if not args.skip_link_calibration:
        calibration = [
            "bash",
            "scripts/deploy/calibrate_cluster_links.sh",
            args.inventory,
            "--run-id",
            run_id,
            "--model-id",
            args.model_id,
            "--timeout-seconds",
            str(args.stage_timeout_seconds),
        ]
        if args.nccl_socket_ifname:
            calibration.extend(["--nccl-socket-ifname", args.nccl_socket_ifname])
        calibration.extend(apply_flag)
        stages.append(("link_calibration", calibration))

    launch = [
        "bash",
        "scripts/deploy/launch_remote.sh",
        args.inventory,
        "--comparison-config",
        args.comparison_config,
        "--strategy",
        args.strategy,
        "--run-id",
        run_id,
        "--model-id",
        args.model_id,
        "--link-cost-profile",
        profile_relative,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.nccl_socket_ifname:
        launch.extend(["--nccl-socket-ifname", args.nccl_socket_ifname])
    if args.apply:
        launch.extend(["--apply", "--wait"])
    stages.append(("launch", launch))
    if not args.no_collect:
        collect = ["bash", "scripts/deploy/collect_remote_logs.sh", args.inventory, "--run-id", run_id]
        validate_results = [
            "bash", "scripts/deploy/summarize_collected_run.sh", args.inventory, "--run-id", run_id
        ]
        if args.apply:
            collect.append("--apply")
            validate_results.append("--apply")
        stages.append(("collect", collect))
        stages.append(("result_validation", validate_results))

    results = []
    launch_attempted = False
    launch_succeeded = False
    collect_command: list[str] | None = None
    result_validation_command: list[str] | None = None
    for name, command in stages:
        if name == "collect":
            collect_command = command
            continue
        if name == "result_validation":
            result_validation_command = command
            continue
        timeout = int(args.timeout_seconds) + 120 if name == "launch" and args.apply else int(args.stage_timeout_seconds)
        result = _run_stage(name, command, log_dir=log_dir, timeout_seconds=timeout)
        results.append(result)
        if name == "launch":
            launch_attempted = True
            launch_succeeded = result["status"] == "PASS"
        if result["status"] != "PASS":
            if name == "launch" and args.apply:
                stop = ["bash", "scripts/deploy/stop_remote_jobs.sh", args.inventory, "--run-id", run_id, "--apply"]
                results.append(
                    _run_stage(
                        "stop_after_launch_failure",
                        stop,
                        log_dir=log_dir,
                        timeout_seconds=int(args.stage_timeout_seconds),
                    )
                )
            break

    # Preserve launch diagnostics even when torchrun fails.  Collection is safe
    # after a successful launch and after the failure cleanup above.
    if collect_command is not None and (launch_succeeded or (launch_attempted and args.apply)):
        collect_result = _run_stage(
            "collect",
            collect_command,
            log_dir=log_dir,
            timeout_seconds=int(args.stage_timeout_seconds),
        )
        results.append(collect_result)
        if collect_result["status"] == "PASS" and result_validation_command is not None:
            results.append(
                _run_stage(
                    "result_validation",
                    result_validation_command,
                    log_dir=log_dir,
                    timeout_seconds=int(args.stage_timeout_seconds),
                )
            )

    required_names = [name for name, _ in stages]
    completed_names = [str(row["name"]) for row in results]
    required_complete = all(name in completed_names for name in required_names)
    status = "PASS" if required_complete and all(
        row["status"] == "PASS" for row in results if row["name"] in required_names
    ) else "FAIL"
    if not args.apply and status == "PASS":
        status = "DRY_RUN_PASS"
    report = {
        "schema_version": "routersense.deploy.pipeline.v2",
        "generated_at": _utc_now(),
        "apply_mode": bool(args.apply),
        "run_id": run_id,
        "inventory": args.inventory,
        "comparison_config": args.comparison_config,
        "strategy": args.strategy,
        "model_id": args.model_id,
        "link_cost_profile": profile_relative,
        "stages": results,
        "status": status,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = _write_human_summary(output_root, report)
    report["human_summary"] = str(summary_path.relative_to(ROOT))
    (output_root / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if status == "FAIL":
        print(
            f"\n[ROUTERSENSE DEPLOYMENT FAILED] Read {summary_path} before taking any action.\n",
            file=sys.stderr,
        )
    print(json.dumps(report, indent=2))
    return 0 if status in {"PASS", "DRY_RUN_PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
