#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime import gpu_environment_snapshot


def main() -> int:
    print(json.dumps(gpu_environment_snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
