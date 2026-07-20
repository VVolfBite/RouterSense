#!/usr/bin/env python3
"""Unified experiment launcher for formal offline and online runs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.experiment_config import build_launch_command, load_run_config, resolve_entrypoint_module


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--apply", action="store_true", default=False)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_run_config(
        config_path=args.config,
        overrides=list(args.override),
        run_id=args.run_id,
        output_dir=args.artifact_root,
    )
    command = build_launch_command(
        config=config,
        config_path=args.config,
        overrides=list(args.override),
        run_id=args.run_id,
        output_dir=args.artifact_root,
    )
    resolved = {
        "config_path": str(Path(args.config).resolve()),
        "run_kind": config.run.kind,
        "run_name": config.run.name,
        "entrypoint_module": resolve_entrypoint_module(config.run.kind),
        "artifact_directory": config.artifact.output_root,
        "observation_profile": config.observation.profile,
        "policy_name": config.online_policy.name,
        "execution_mode": config.execution.mode,
        "control_mode": config.runtime.control_mode,
        "topology": config.topology.launcher.__dict__,
        "torchrun_command": command,
    }
    print("Resolved config:")
    for key, value in resolved.items():
        if key == "torchrun_command":
            print(f"{key}: {shlex.join(value)}")
        else:
            print(f"{key}: {value}")
    if not args.apply:
        return 0
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
