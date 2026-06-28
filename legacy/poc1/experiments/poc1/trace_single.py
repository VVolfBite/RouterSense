#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(ROOT / "outputs" / "poc1_trace_single")])

runpy.run_path(str(ROOT / "scripts" / "trace_one_token.py"), run_name="__main__")
