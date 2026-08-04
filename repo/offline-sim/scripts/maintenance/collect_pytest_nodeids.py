#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest


class ExactCollector:
    def __init__(self) -> None:
        self.nodeids: list[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.nodeids = [item.nodeid for item in session.items]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect exact pytest session.items node IDs")
    parser.add_argument("root", nargs="?", default="tests")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    plugin = ExactCollector()
    code = int(pytest.main(["--collect-only", "-q", str(args.root)], plugins=[plugin]))
    payload = {
        "schema_version": "RS_SIM_PYTEST_NODEIDS",
        "root": str(args.root),
        "collection_return_code": code,
        "item_count": len(plugin.nodeids),
        "nodeids": plugin.nodeids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "nodeids"}, sort_keys=True))
    return 0 if code == 0 else code


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(code))
