#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

defaults = {
    "--output-dir": str(ROOT / "outputs" / "poc1_server_trace_batch"),
    "--max-samples": "48",
    "--layers": "0,4,8,12,15",
    "--comment": "server-ready trace batch entry; larger sample/layer sweep without runtime or ablation",
}

for key, value in defaults.items():
    if key not in sys.argv:
        sys.argv.extend([key, value])

runpy.run_path(str(ROOT / "scripts" / "summarize_trace_batch.py"), run_name="__main__")
