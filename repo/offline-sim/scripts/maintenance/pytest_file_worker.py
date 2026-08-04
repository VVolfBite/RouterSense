#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys
import traceback

# The worker is normally launched as ``python tools/pytest_file_worker.py``.
# In that mode Python puts ``tools/`` rather than the repository root on
# ``sys.path``.  Integration tests are allowed to import repository-owned tools
# (for example ``tools.analyze_p12_causal_metrics``), so restore the same import
# contract as a normal pytest invocation from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pytest


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: pytest_file_worker.py TEST_FILE", file=sys.stderr)
        return 2
    return int(pytest.main(["-q", args[0]]))


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(int(code))
