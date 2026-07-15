#!/usr/bin/env python3
"""Run online phase-local strategies and aggregate comparison metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.experiments_support.strategy_comparison_runner import run_strategy_comparison


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_strategy_comparison(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
