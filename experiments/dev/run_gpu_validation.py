#!/usr/bin/env python3
"""Deprecated compatibility wrapper for the unified validation entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("b2", "c2", "a2"), required=True)
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    proc = __import__("subprocess").run(
        [sys.executable, "experiments/dev/run_validation.py", "--suite", str(args.suite), *list(args.passthrough)],
        cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT / "src")},
        check=False,
    )
    raise SystemExit(int(proc.returncode))


if __name__ == "__main__":
    main()
