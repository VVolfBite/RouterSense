#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.evaluation import run_dc_asymmetry_analysis, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run dispatch/combine asymmetry analysis on existing POC-line1 traces.")
    parser.add_argument("--trace-jsonl", type=str, required=True)
    parser.add_argument("--placement", type=str, default="round_robin", choices=["round_robin", "skewed"])
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--pairwise-results-json", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "dc_asymmetry_report"),
    )
    args = parser.parse_args(argv)

    report = run_dc_asymmetry_analysis(
        trace_jsonl=args.trace_jsonl,
        placement=args.placement,
        num_gpus=args.num_gpus,
        pairwise_results_json=args.pairwise_results_json,
        sample_limit=args.sample_limit,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, payload in report.items():
        write_json(out / f"{key}.json", payload)
    print(json.dumps(report["go_no_go"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
