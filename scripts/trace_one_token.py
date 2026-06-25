#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc1.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace one token")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "poc1.yaml"))
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "poc1_trace_single"))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    argv = ["--config", args.config]
    if args.text is not None:
        argv.extend(["--text", args.text])
    if args.layer is not None:
        argv.extend(["--layer", args.layer])
    if args.model_id is not None:
        argv.extend(["--model-id", args.model_id])
    if args.output_dir is not None:
        argv.extend(["--output-dir", args.output_dir])
    if args.mock:
        argv.append("--mock")
    raise SystemExit(main(argv, command="trace"))
