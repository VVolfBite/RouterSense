#!/usr/bin/env python3
"""Run the related-work style core study on extracted FATE traffic traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rs.runtime.offline.related_work_core_study import (
    DEFAULT_POLICIES,
    discover_trace_roots,
    run_related_work_core_study,
    write_study_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--max-waves", type=int, default=4096)
    args = parser.parse_args()
    policies = tuple(item.strip() for item in str(args.policies).split(",") if item.strip())
    payload = run_related_work_core_study(
        trace_roots=discover_trace_roots(args.trace_root),
        policy_names=policies,
        max_waves=int(args.max_waves),
    )
    write_study_artifacts(payload, args.output_dir)
    print(json.dumps({"scope": payload["scope"], "summary": payload["summary"]}, ensure_ascii=False, indent=2))
    return 0 if all(int(row["invalid_count"]) == 0 for row in payload["summary"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
