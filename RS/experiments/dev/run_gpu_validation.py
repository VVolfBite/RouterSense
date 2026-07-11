#!/usr/bin/env python3
"""Single validation entrypoint for GPU B2/C2/A2 suites."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()


SUITE_TO_SCRIPT = {
    "b2": "experiments/distributed/run_gpu_b2_lifecycle.py",
    "c2": "experiments/distributed/run_gpu_c2_async_correctness.py",
    "a2": "experiments/distributed/run_gpu_a2_strategy_compare.py",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITE_TO_SCRIPT), required=True)
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    script = SUITE_TO_SCRIPT[str(args.suite)]
    proc = subprocess.run(
        [sys.executable, script, *list(args.passthrough)],
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
    )
    raise SystemExit(int(proc.returncode))


if __name__ == "__main__":
    main()
