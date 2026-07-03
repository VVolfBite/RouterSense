#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime import gpu_environment_snapshot


def main() -> int:
    print(json.dumps(gpu_environment_snapshot(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
