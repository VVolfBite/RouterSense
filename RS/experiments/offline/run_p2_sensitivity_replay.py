#!/usr/bin/env python3
"""Offline replay for P2-signal sensitivity on safe-U policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()
from rs.runtime.offline.p2_sensitivity import render_markdown, run_p2_sensitivity_replay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    return parser.parse_args()
def main() -> None:
    args = _parse_args()
    payload = run_p2_sensitivity_replay(fixture_dir=Path(args.fixture_dir))
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_summary_md).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"output_summary": args.output_summary, "row_count": len(payload["rows"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
