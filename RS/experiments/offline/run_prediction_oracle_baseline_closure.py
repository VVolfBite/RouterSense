#!/usr/bin/env python3
"""Thin wrapper for the formal prediction/oracle/baseline closure entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rs.runtime.offline.prediction_oracle_baseline_closure import main


if __name__ == "__main__":
    main()
