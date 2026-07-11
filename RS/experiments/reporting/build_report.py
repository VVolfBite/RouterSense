#!/usr/bin/env python3
"""Unified structured report builder for official experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.reporting.offline_report import build_offline_report
from rs.reporting.performance_report import build_performance_report
from rs.reporting.runtime_audit_report import build_runtime_audit_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--report-type", choices=("offline", "c2", "a2", "comparison", "runtime_audit"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = (ROOT / str(args.input)).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if args.report_type == "offline":
        bundle = build_offline_report(run_dir)
    elif args.report_type == "runtime_audit":
        bundle = build_runtime_audit_report(run_dir)
    else:
        bundle = build_performance_report(run_dir, report_type=str(args.report_type))
    (reports_dir / f"{bundle.report_type}.md").write_text(bundle.markdown, encoding="utf-8")
    (reports_dir / f"{bundle.report_type}.json").write_text(json.dumps(bundle.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_type": bundle.report_type, "output_dir": str(reports_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
