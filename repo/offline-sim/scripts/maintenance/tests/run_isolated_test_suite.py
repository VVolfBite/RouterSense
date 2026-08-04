#!/usr/bin/env python3
from __future__ import annotations

"""Legacy compatibility wrapper for the bounded sequential pytest runner."""

import argparse
from pathlib import Path

from run_bounded_pytest_items import main as bounded_main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper; use tools/run_bounded_pytest.py directly"
    )
    parser.add_argument("--root", default="tests")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", default="artifacts/bounded_pytest/summary.json")
    args = parser.parse_args()
    if args.jobs != 1:
        raise SystemExit("parallel pytest execution is disabled; --jobs must be 1")
    output_dir = Path(args.output).parent
    return bounded_main([
        args.root,
        "--item-timeout-seconds", str(args.timeout_seconds),
        "--output-dir", str(output_dir),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
