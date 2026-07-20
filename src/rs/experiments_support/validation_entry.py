#!/usr/bin/env python3
"""Unified validation entrypoint for offline smoke, Gloo, and GPU validation suites."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


SUITE_TO_CMD = {
    "offline-smoke": [
        sys.executable,
        "experiments/run_offline_replay.py",
        "--config",
        "tests/fixtures/configs/minimal_offline.yaml",
        "--output-dir",
        "outputs/validation/offline_smoke",
    ],
    "config": [sys.executable, "-m", "pytest", "-q", "tests/contract/test_config_normalization.py"],
    "catalog": [sys.executable, "-m", "pytest", "-q", "tests/contract/test_replay_unified_closure.py"],
    "compiler": [sys.executable, "-m", "pytest", "-q", "tests/contract/test_unified_interface_refactor.py"],
    "transport": [sys.executable, "-m", "pytest", "-q", "tests/contract/megatron_ep/test_async_p2p_runtime_closure.py"],
    "gloo": [sys.executable, "experiments/distributed/run_stage3_runtime_integrated_gloo_gate_lowmem.py"],
    "guard-faults": [sys.executable, "experiments/distributed/run_stage3_guard_fault_injection_gate.py"],
    "b2": [
        sys.executable,
        "experiments/distributed/run_gpu_b2_lifecycle.py",
        "--config",
        "configs/official/online_async_release.yaml",
        "--output-dir",
        "outputs/validation/b2",
        "--strategy",
        "routersense_joint_zero_hint_async_p2p",
        "--profile",
        "execution",
        "--selected-layers",
        "selected",
        "--world-size",
        "4",
        "--dry-run",
    ],
    "c2": [
        sys.executable,
        "experiments/distributed/run_gpu_c2_async_correctness.py",
        "--config",
        "configs/official/gpu_c2_correctness.yaml",
        "--output-dir",
        "outputs/validation/c2",
        "--dry-run",
    ],
    "a2": [
        sys.executable,
        "experiments/distributed/run_gpu_a2_strategy_compare.py",
        "--config",
        "configs/official/gpu_a2_performance.yaml",
        "--output-dir",
        "outputs/validation/a2",
        "--dry-run",
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITE_TO_CMD), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Force dry-run mode for suites that support real execution.")
    mode.add_argument("--execute", action="store_true", help="Force real execution for suites that support it.")
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough
    return args


def main() -> None:
    args = _parse_args()
    cmd = [*SUITE_TO_CMD[str(args.suite)]]
    supports_execution_toggle = str(args.suite) in {"b2", "c2", "a2"}
    if supports_execution_toggle:
        if args.execute:
            cmd = [item for item in cmd if item != "--dry-run"]
        elif args.dry_run or not args.execute:
            if "--dry-run" not in cmd:
                cmd.append("--dry-run")
    elif args.execute:
        raise SystemExit(f"--execute is not supported for suite={args.suite}")
    elif args.dry_run:
        raise SystemExit(f"--dry-run override is not supported for suite={args.suite}")
    cmd.extend(list(args.passthrough))
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT / "src") if not existing else f"{ROOT / 'src'}:{existing}"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    raise SystemExit(int(proc.returncode))


if __name__ == "__main__":
    main()
