#!/usr/bin/env python3
"""Estimate whether prediction/planning control cost can be hidden offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()
from rs.runtime.offline.planning_hiding import estimate_planning_hiding_window


def main() -> None:
    args = _parse_args()
    result = estimate_planning_hiding_window(
        planning_summary_path=Path(args.planning_summary),
        observed_forward_us=float(args.observed_forward_us),
        moe_layer_count=int(args.moe_layer_count),
        observed_moe_interval_us=float(args.observed_moe_interval_us),
    )
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
