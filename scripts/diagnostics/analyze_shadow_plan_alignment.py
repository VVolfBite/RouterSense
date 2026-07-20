#!/usr/bin/env python3
"""Analyze prepared-window shadow plans against actual scheduled execution artifacts."""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.reporting.shadow_plan_analysis import analyze_rank_artifacts
from rs.runtime.online.megatron_ep.observation import write_json


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    report = analyze_rank_artifacts(run_dir, rank=int(args.rank))
    if args.output:
        write_json(Path(args.output), report)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
