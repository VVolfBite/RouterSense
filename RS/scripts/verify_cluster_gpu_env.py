#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense.runtime import gpu_environment_snapshot


def main() -> int:
    print(json.dumps(gpu_environment_snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
