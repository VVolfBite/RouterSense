#!/usr/bin/env python3
"""Small-instance oracle gap replay and real-fixture oracle proxy summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.offline.oracle_gap_replay import run_oracle_gap_replay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--policies", nargs="*", default=None)
    return parser.parse_args()
def main() -> None:
    args = _parse_args()
    payload = run_oracle_gap_replay(
        fixture_dir=Path(args.fixture_dir),
        small_only=bool(args.small_only),
        policies=args.policies,
    )
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
