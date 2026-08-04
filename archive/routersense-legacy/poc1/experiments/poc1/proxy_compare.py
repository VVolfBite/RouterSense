#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if "--input-dir" not in sys.argv:
    sys.argv.extend(["--input-dir", str(ROOT / "outputs" / "poc1_trace_batch")])
if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(ROOT / "outputs" / "poc1_proxy_compare")])

runpy.run_path(str(ROOT / "scripts" / "analyze_proxy_scores.py"), run_name="__main__")
