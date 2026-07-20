#!/usr/bin/env python3
"""Run the formal unified tiny O_local/O_joint exact suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ensure_src_on_path()

from rs.runtime.offline.exact_oracle_suite import run_exact_scope_suite


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-count", type=int, default=32)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_exact_scope_suite(int(args.instance_count))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
