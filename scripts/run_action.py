#!/usr/bin/env python3
"""Legacy action dispatcher.

Prefer experiment/poc1/* entrypoints for current POC1 work.
"""
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
    parser = argparse.ArgumentParser(description="Run a RouteSense PoC1 action")
    parser.add_argument("action", choices=["inspect", "trace", "ablate"])
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "poc1.yaml"))
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    argv = ["--config", args.config]
    if args.text is not None:
        argv.extend(["--text", args.text])
    if args.layer is not None:
        argv.extend(["--layer", args.layer])
    if args.rank is not None:
        argv.extend(["--rank", str(args.rank)])
    if args.model_id is not None:
        argv.extend(["--model-id", args.model_id])
    output_dir = args.output_dir
    if output_dir is None and args.action == "trace":
        output_dir = str(ROOT / "outputs" / "poc1_trace_single")
    if output_dir is not None:
        argv.extend(["--output-dir", output_dir])
    if args.mock:
        argv.append("--mock")

    raise SystemExit(main(argv, command=args.action))
