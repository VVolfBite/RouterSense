#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc2.eval import build_eval_summary


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    input_path = Path(argv[0]) if argv else ROOT / "outputs" / "poc2_single_node_eval" / "summary.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    summary = build_eval_summary(payload.get("runs", []))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
