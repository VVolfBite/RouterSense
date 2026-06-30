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

from routesense.evaluation import analyze_cross_layer_correlation, load_trace_jsonl, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run POC-line1 cross-layer correlation analysis.")
    parser.add_argument("--trace-jsonl", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "poc_line1" / "cross_layer_report"),
    )
    args = parser.parse_args(argv)

    records = load_trace_jsonl(args.trace_jsonl)
    report = analyze_cross_layer_correlation(records)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in report.items():
        write_json(out / f"{name}.json", payload)
    print(json.dumps(report["gate1_decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
