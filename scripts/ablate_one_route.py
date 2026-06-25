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
    parser = argparse.ArgumentParser(description="Ablate one route")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "poc1.yaml"))
    args = parser.parse_args()
    argv = ["--config", args.config]
    if args.text is not None:
        argv.extend(["--text", args.text])
    if args.layer is not None:
        argv.extend(["--layer", args.layer])
    if args.rank is not None:
        argv.extend(["--rank", str(args.rank)])
    raise SystemExit(main(argv, command="ablate"))
