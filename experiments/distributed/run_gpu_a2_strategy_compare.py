#!/usr/bin/env python3
"""Thin CLI for the formal 4-GPU A2 strategy comparison runner."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rs.experiments_support.gpu_a2_strategy_compare import main


if __name__ == "__main__":
    raise SystemExit(main())
